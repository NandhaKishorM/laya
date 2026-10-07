# NixOS module: run laya-serve (Laya's Jev-compatible /v1/systemone HTTP API)
# as a hardened systemd service with CUDA access.
#
# The package is built from the host's own `pkgs` via ./package.nix (which pulls
# the prebuilt CUDA torch, so the host needs `allowUnfree`). This deliberately
# avoids an overlay, so the module also works on hosts that inject `pkgs` via
# specialArgs and ignore module-level `nixpkgs.overlays`.
{ config, lib, pkgs, ... }:
let
  cfg = config.services.laya-serve;

  startScript = pkgs.writeShellScript "laya-serve-start" ''
    set -eu
    ${lib.optionalString (cfg.apiKeyFile != null) ''
      export LAYA_API_KEY="$(cat "$CREDENTIALS_DIRECTORY/apikey")"
    ''}
    exec ${cfg.package}/bin/laya-serve
  '';
in
{
  options.services.laya-serve = {
    enable = lib.mkEnableOption "Laya System-1 decision server (TypeSafe Jev-compatible HTTP API)";

    package = lib.mkOption {
      type = lib.types.package;
      default = (pkgs.callPackage ./package.nix { }).laya-serve;
      defaultText = lib.literalExpression "(pkgs.callPackage ./package.nix { }).laya-serve";
      description = ''
        The laya-serve package to run. Built from the host's own `pkgs` (needs
        `allowUnfree` for the prebuilt CUDA torch), so it works regardless of how
        the host provides `pkgs`.
      '';
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Address to bind. Use 0.0.0.0 to serve the LAN/Tailscale net.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "TCP port to listen on.";
    };

    rootPath = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "/laya";
      description = ''
        Public URL prefix used by FastAPI behind a reverse proxy (sets
        `LAYA_ROOT_PATH`). The proxy should strip this prefix before forwarding
        requests. null leaves the root path empty.
      '';
    };

    device = lib.mkOption {
      type = lib.types.str;
      default = "cuda";
      example = "cpu";
      description = "torch device for every checkpoint (cuda, cpu, cuda:0, ...).";
    };

    cudaAmp = lib.mkOption {
      type = lib.types.nullOr (lib.types.strMatching "[a-zA-Z0-9_-]+");
      default = null;
      example = "fp16";
      description = ''
        Autocast dtype on CUDA, overriding the checkpoint's own `amp_dtype`
        (sets `LAYA_CUDA_AMP`). Which tokens mean something is decided by
        `laya.agent`, not here: `tests/test_packaging.py` reads them out of the
        runtime and fails if this text drifts from it, which is why there is no
        `enum` on this option. The recognised spellings are `fp16`, `float16`,
        `bf16` and `bfloat16` — the first two name one dtype and the last two
        another. Anything else is ignored rather than an error, and the
        checkpoint's own choice stays in force.

        The type allows any single token but refuses whitespace, which is not
        style: the runtime lower-cases the value and then compares it without
        trimming, so a trailing space keeps full precision and says nothing.
        null leaves every CUDA checkpoint on its own `amp_dtype`.
      '';
    };

    cpuAmp = lib.mkOption {
      type = lib.types.nullOr (lib.types.strMatching "[a-zA-Z0-9_-]+");
      default = null;
      example = "bf16";
      description = ''
        Run CPU inference under autocast (sets `LAYA_CPU_AMP`). CPU is the one
        device where the runtime does not do this by default: reduced precision
        on CPU only pays where the hardware has a native BF16 path. The
        accepted spellings are `bf16` and `bfloat16`, and nothing else —
        deliberately narrower than `cudaAmp`, which additionally takes a
        half-precision spelling that means nothing on CPU. A value the runtime
        does not recognise leaves CPU in fp32 rather than raising, and
        `tests/test_packaging.py` derives both device sets from `laya.agent`,
        so this sentence is checked in both directions rather than trusted.

        On a host without a native BF16 path this costs several times the fp32
        latency for the same decision: measured at 137 ms -> 744 ms on one
        single-row request, answer unchanged. Set it only where the hardware is
        known to want it. null keeps CPU at full precision.
      '';
    };

    mpsAmpMinRows = lib.mkOption {
      type = lib.types.nullOr lib.types.ints.positive;
      default = null;
      example = 1;
      description = ''
        Smallest batch that runs under fp16 autocast on MPS (sets
        `LAYA_MPS_AMP_MIN_ROWS`). MPS autocast is decided per call rather than
        once at load: casting costs more than the matmul saves on a single
        small row, and starts to win once a batch has several, so the runtime
        holds short requests in fp32 and long ones in fp16. This option moves
        that crossover for the whole service.

        There is no value that is simply better, which is why the default is
        null and not a number. Measured on an Apple-silicon host with the
        `english` checkpoint: forcing the threshold to 1 made a one-row request
        42% slower (31.8 ms -> 45.1 ms), while pushing it past the largest
        batch the service sees made an eight-row batch 71% slower
        (83.5 ms -> 142.7 ms) — the two settings trade the same two cases
        against each other. The reported answer was unchanged either way.

        `ints.positive` rather than `ints.unsigned`: the runtime clamps a value
        below 1 up to 1, so this refuses at evaluation time the one input the
        service would otherwise reinterpret. null leaves the runtime's own
        threshold in place.
      '';
    };

    models = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ "english" "multilingual" "typed-decisions" ];
      description = ''
        Checkpoints to preload at startup. All three fit comfortably in a 24 GB
        card (~1.16B params total), so the default keeps every one hot and makes
        language routing free.

        The names are validated by laya rather than by this module: the list is
        joined into `LAYA_MODELS`, and the server normalises every entry with
        `laya.router.normalise_name`, which accepts `english`, `multilingual` and
        `typed-decisions` plus the aliases `en`, `laya`, `default`, `multi`, `ml`,
        `laya-multilingual`, `typed`, `typed_decisions`, `laya-typed-decisions` and
        `decisions`. An unknown name stops the service at startup with an error
        listing all of them. An `enum` here could only copy that list and fall
        behind it, refusing a spelling the server accepts.
      '';
    };

    preload = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Build the checkpoints at startup rather than lazily on first request.";
    };

    idleUnloadSeconds = lib.mkOption {
      type = lib.types.nullOr lib.types.ints.unsigned;
      default = null;
      example = 300;
      description = ''
        Unload resident checkpoints after this many idle seconds (sets
        `LAYA_IDLE_UNLOAD_SECONDS`). The next request pays a cold load.
        Zero disables unloading; null leaves the server's default, also off.
        Device caches are released, but the process allocator may retain RAM pages.
      '';
    };

    threads = lib.mkOption {
      type = lib.types.nullOr lib.types.ints.positive;
      default = null;
      example = 16;
      description = ''
        Cap torch intra-op threads for CPU inference (sets LAYA_THREADS and
        OMP_NUM_THREADS). Ignored in practice on CUDA. Keep this at or below the
        host's *physical* core count — oversubscribing the logical/hyperthread
        count is a large latency regression. For single-request latency, a value
        below the core count (e.g. 8-16) is often fastest; for batched
        throughput, the physical core count is best. null leaves torch's default.
      '';
    };

    logLevel = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "warning";
      description = ''
        uvicorn's log level (sets `LAYA_LOG_LEVEL`), one of `critical`, `error`,
        `warning`, `info`, `debug`, `trace`. `info`, which the server uses when
        this is unset, logs a line per request, so on a busy decision endpoint it
        is most of the unit's journal traffic; `warning` keeps startup and errors.
        Not an `enum` over those names: uvicorn checks the value, and a copy here
        could only fall behind it. null leaves the server's own default.
      '';
    };

    maxConcurrent = lib.mkOption {
      type = lib.types.nullOr lib.types.ints.positive;
      default = null;
      example = 4;
      description = ''
        Cap on requests admitted at once (sets `LAYA_MAX_CONCURRENT`). A request
        that arrives with every slot taken is refused with HTTP 503 right away
        rather than queued, which is what keeps the bodies held in memory
        bounded. Lower it on a host where a queue of forwards is worse
        than a refusal — measured on the `english` checkpoint, CPU, 16
        simultaneous requests: the median accepted request answered in 64 ms at a
        cap of 1 and 542 ms at 16, and 15 of 16 were refused in under 0.1 ms at
        that cap against none at 16. So the knob trades the latency of the
        requests that get through against the number turned away, and a host
        needs to be able to pick. null leaves the server's own default.
      '';
    };

    maxLoaded = lib.mkOption {
      type = lib.types.nullOr lib.types.ints.positive;
      default = null;
      example = 3;
      description = ''
        Checkpoints kept resident at once (sets `LAYA_MAX_LOADED`). The Router
        keeps two by default, the number automatic routing chooses between;
        with `autoTaskDetection` a third one becomes reachable on demand, and a
        cap below what routing chooses rebuilds a checkpoint on every switch.
        null leaves the Router's own default.
      '';
    };

    maxTokenBudget = lib.mkOption {
      type = lib.types.nullOr lib.types.ints.positive;
      default = null;
      example = 4096;
      description = ''
        Ceiling on the per-request `max_len` / `head_max_len` a client may ask
        for (sets `LAYA_MAX_TOKEN_BUDGET`); a larger value is refused with 422.
        null leaves the server's own default.
      '';
    };

    maxBatchTokens = lib.mkOption {
      type = lib.types.nullOr lib.types.ints.positive;
      default = null;
      example = 262144;
      description = ''
        Tokens one /v1/systemone/batch FORWARD PASS may collate -- states x
        questions x the row width (sets `LAYA_MAX_BATCH_TOKENS`). A larger batch
        is split across several passes, not refused. null leaves the server's own
        default.
      '';
    };

    revision = lib.mkOption {
      type = lib.types.nullOr (lib.types.strMatching "[A-Za-z0-9._/-]+");
      default = null;
      example = "reviewed";
      description = ''
        Hub commit, branch or tag every checkpoint is downloaded at (sets
        `LAYA_REVISION`), or `reviewed` for the reviewed commit SHAs laya ships.
        null keeps huggingface_hub's default revision and any existing cache.
      '';
    };

    autoTaskDetection = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Let the router auto-select the typed-decisions checkpoint when question ids match its workflows.";
    };

    jevStrict = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Project every `/v1/systemone` answer onto the strict Jev wire contract
        (sets `LAYA_JEV_STRICT`): only the fields the vendor Jev API itself
        returns — `model`/`answers`/`usage` at the root, the contracted keys per
        answer, and a two-key `usage`. Nothing is recomputed; every value is the
        one the result already carries. A client that validates the response
        against the contract with no extra fields rejects the full payload, so
        this makes the server usable with such clients; OpenClaw's
        `@openclaw/typesafe` decision provider is one. Default off, full payload
        unchanged.
      '';
    };

    defaultModel = lib.mkOption {
      type = lib.types.nullOr (lib.types.strMatching "[A-Za-z0-9._/-]+");
      default = null;
      example = "multilingual";
      description = ''
        Checkpoint a state with no language evidence falls back to (sets
        `LAYA_DEFAULT_MODEL`): text with no letters at all, or Latin script too
        short to identify a language. Set it to `multilingual` when most traffic
        on this host is not English. Text the detector can place is routed on
        what it detects, so this is a fallback and not a pin. The spellings
        accepted, and the aliases among them, are laya's own -- the server
        checks the value against them at startup and refuses to start on a name
        it does not know, so this module deliberately lists none. null leaves
        laya's default, which is `english`.
      '';
    };

    apiKeyFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = lib.literalExpression "config.age.secrets.laya-api-key.path";
      description = ''
        Path to a file containing a bearer token. When set, clients must send
        `Authorization: Bearer <token>`. Read via systemd LoadCredential, so it
        never lands in the store or the unit's environment.
      '';
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open `port` in the firewall.";
    };

    stateDirectory = lib.mkOption {
      type = lib.types.str;
      default = "laya-serve";
      description = "Name under /var/lib for the Hugging Face weight cache (HF_HOME).";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.laya-serve = {
      description = "Laya System-1 decision server (Jev-compatible)";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];

      environment = {
        LAYA_HOST = cfg.host;
        LAYA_PORT = toString cfg.port;
        LAYA_DEVICE = cfg.device;
        LAYA_PRELOAD = if cfg.preload then "1" else "0";
        LAYA_MODELS = lib.concatStringsSep "," cfg.models;
        LAYA_AUTO_TASK = if cfg.autoTaskDetection then "1" else "0";
        LAYA_JEV_STRICT = if cfg.jevStrict then "1" else "0";
      } // lib.optionalAttrs (cfg.threads != null) {
        LAYA_THREADS = toString cfg.threads;
        OMP_NUM_THREADS = toString cfg.threads;
      } // lib.optionalAttrs (cfg.idleUnloadSeconds != null) {
        LAYA_IDLE_UNLOAD_SECONDS = toString cfg.idleUnloadSeconds;
      } // lib.optionalAttrs (cfg.rootPath != null) {
        LAYA_ROOT_PATH = cfg.rootPath;
      } // lib.optionalAttrs (cfg.logLevel != null) {
        LAYA_LOG_LEVEL = cfg.logLevel;
      } // lib.optionalAttrs (cfg.maxConcurrent != null) {
        LAYA_MAX_CONCURRENT = toString cfg.maxConcurrent;
      } // lib.optionalAttrs (cfg.cudaAmp != null) {
        LAYA_CUDA_AMP = cfg.cudaAmp;
      } // lib.optionalAttrs (cfg.cpuAmp != null) {
        LAYA_CPU_AMP = cfg.cpuAmp;
      } // lib.optionalAttrs (cfg.mpsAmpMinRows != null) {
        LAYA_MPS_AMP_MIN_ROWS = toString cfg.mpsAmpMinRows;
      } // lib.optionalAttrs (cfg.maxLoaded != null) {
        LAYA_MAX_LOADED = toString cfg.maxLoaded;
      } // lib.optionalAttrs (cfg.maxTokenBudget != null) {
        LAYA_MAX_TOKEN_BUDGET = toString cfg.maxTokenBudget;
      } // lib.optionalAttrs (cfg.maxBatchTokens != null) {
        LAYA_MAX_BATCH_TOKENS = toString cfg.maxBatchTokens;
      } // lib.optionalAttrs (cfg.revision != null) {
        LAYA_REVISION = cfg.revision;
      } // lib.optionalAttrs (cfg.defaultModel != null) {
        LAYA_DEFAULT_MODEL = cfg.defaultModel;
      } // {
        HF_HOME = "/var/lib/${cfg.stateDirectory}/huggingface";
        # torch-bin bundles its own CUDA runtime but still needs the host
        # driver's libcuda.so.1 / libnvidia-ml.so, which NixOS exposes here.
        LD_LIBRARY_PATH = "/run/opengl-driver/lib";
      };

      serviceConfig = {
        ExecStart = startScript;
        Restart = "on-failure";
        RestartSec = 5;
        # First start downloads ~1.2 GB of weights before it listens.
        TimeoutStartSec = "600";

        DynamicUser = true;
        StateDirectory = cfg.stateDirectory;

        # GPU: keep the nvidia device nodes visible to the sandbox.
        PrivateDevices = false;
        DeviceAllow = [
          "/dev/nvidia0 rw"
          "/dev/nvidiactl rw"
          "/dev/nvidia-uvm rw"
          "/dev/nvidia-uvm-tools rw"
          "/dev/nvidia-modeset rw"
        ];

        # Hardening (kept compatible with CUDA device access).
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        ProtectControlGroups = true;
        ProtectKernelModules = true;
        RestrictNamespaces = true;
        RestrictSUIDSGID = true;
        LockPersonality = true;
      } // lib.optionalAttrs (cfg.apiKeyFile != null) {
        LoadCredential = [ "apikey:${cfg.apiKeyFile}" ];
      };
    };

    networking.firewall.allowedTCPPorts = lib.optional cfg.openFirewall cfg.port;
  };
}
