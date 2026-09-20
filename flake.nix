{
  description = "laya + laya-serve: a self-hosted, TypeSafe Jev-compatible System-1 decision server";

  inputs = {
    # Pinned to the same rev missionctrl-infra uses, so hq's binary cache
    # (cache.nixos.org + missionctrl.cachix.org) hits instead of rebuilding torch.
    nixpkgs.url = "github:NixOS/nixpkgs/ccad53cd79cf4cf3bc338805d007d68565e75bda";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      # Overlay exposing `laya` + `laya-serve`, built from ./nix/package.nix.
      overlay = final: prev:
        let p = final.callPackage ./nix/package.nix { };
        in { inherit (p) laya laya-serve; };
    in
    {
      overlays.default = overlay;

      # The module builds its package from the host's own `pkgs` (see
      # nix/laya-serve.nix), so it works even on hosts that inject `pkgs` via
      # specialArgs and ignore module-level `nixpkgs.overlays` — no overlay
      # required here.
      nixosModules.default = ./nix/laya-serve.nix;
      nixosModules.laya-serve = ./nix/laya-serve.nix;
    }
    // flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true; # torch-bin bundles CUDA (unfree)
          overlays = [ overlay ];
        };
      in
      {
        packages = {
          default = pkgs.laya-serve;
          laya-serve = pkgs.laya-serve;
          laya = pkgs.laya;
        };

        devShells.default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (ps: [
              ps.torch-bin
              ps.transformers
              ps.safetensors
              ps.huggingface-hub
              ps.numpy
              ps.fastapi
              ps.uvicorn
              ps.pytest
              ps.httpx # fastapi TestClient
            ]))
          ];
          shellHook = ''
            export PYTHONPATH="$PWD:$PYTHONPATH"
            # torch-bin's CUDA needs the host NVIDIA userspace driver.
            export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            echo "laya dev shell — python $(python --version 2>&1 | cut -d' ' -f2), torch $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
          '';
        };
      });
}
