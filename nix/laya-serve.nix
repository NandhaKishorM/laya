# NixOS module: run laya-serve (Laya's Jev-compatible /v1/systemone HTTP API)
# as a hardened systemd service with CUDA access.
#
# The package comes from this flake's overlay (`pkgs.laya-serve`). The flake's
# `nixosModules.default` applies that overlay for you; if you import this file
# directly, either apply `overlays.default` yourself or set
# `services.laya-serve.package` explicitly.
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
      default = pkgs.laya-serve;
      defaultText = lib.literalExpression "pkgs.laya-serve";
      description = "The laya-serve package to run.";
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

    device = lib.mkOption {
      type = lib.types.str;
      default = "cuda";
      example = "cpu";
      description = "torch device for every checkpoint (cuda, cpu, cuda:0, ...).";
    };

    models = lib.mkOption {
      type = lib.types.listOf (lib.types.enum [ "english" "multilingual" "typed-decisions" ]);
      default = [ "english" "multilingual" "typed-decisions" ];
      description = ''
        Checkpoints to preload at startup. All three fit comfortably in a 24 GB
        card (~1.16B params total), so the default keeps every one hot and makes
        language routing free.
      '';
    };

    preload = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Build the checkpoints at startup rather than lazily on first request.";
    };

    autoTaskDetection = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Let the router auto-select the typed-decisions checkpoint when question ids match its workflows.";
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
