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
      # Overlay: the `laya` python package + a `laya-serve` runner. Portable to
      # any nixpkgs (e.g. missionctrl-infra's system pkgs).
      overlay = final: prev:
        let
          py = final.python3Packages;
          laya = py.buildPythonPackage {
            pname = "laya";
            version = "0.3.4";
            src = ./.;
            format = "setuptools";
            propagatedBuildInputs = [
              py.torch-bin # prebuilt CUDA wheel — no source build
              py.transformers
              py.safetensors
              py.huggingface-hub
              py.numpy
            ];
            # Every test loads a checkpoint from the Hub -> needs network + a GPU.
            doCheck = false;
            # serve.py defers its fastapi/uvicorn imports, so this stays honest
            # without dragging the web stack into the base library.
            pythonImportsCheck = [ "laya" "laya.serve" ];
          };
          # The runner bundles the server deps so the unit needs nothing else.
          pyEnv = final.python3.withPackages (ps: [ laya ps.fastapi ps.uvicorn ]);
        in
        {
          inherit laya;
          laya-serve = final.writeShellScriptBin "laya-serve" ''
            exec ${pyEnv}/bin/python -m laya.serve "$@"
          '';
        };
    in
    {
      overlays.default = overlay;

      # Import into a NixOS host; this also applies the overlay so
      # `pkgs.laya-serve` resolves.
      nixosModules.default = { ... }: {
        imports = [ ./nix/laya-serve.nix ];
        nixpkgs.overlays = [ self.overlays.default ];
      };
      nixosModules.laya-serve = self.nixosModules.default;
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
