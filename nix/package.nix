# The laya python package + a laya-serve runner, built from whatever `pkgs` is
# in scope. Factored out so both the flake's packages/overlay AND the NixOS
# module can build it the same way -- the module cannot rely on an overlay,
# because hosts that inject `pkgs` via specialArgs ignore module-level
# `nixpkgs.overlays`.
{ lib, python3, python3Packages, writeShellScriptBin }:
let
  laya = python3Packages.buildPythonPackage {
    pname = "laya";
    version = "0.3.4";
    src = ../.;
    format = "setuptools";
    propagatedBuildInputs = with python3Packages; [
      torch-bin # prebuilt CUDA wheel -- no source build
      transformers
      safetensors
      huggingface-hub
      numpy
    ];
    # Every test loads a checkpoint from the Hub -> needs network + a GPU.
    doCheck = false;
    # serve.py defers its fastapi/uvicorn imports, so this stays honest without
    # dragging the web stack into the base library.
    pythonImportsCheck = [ "laya" "laya.serve" ];
  };
  pyEnv = python3.withPackages (ps: [ laya ps.fastapi ps.uvicorn ]);
in
{
  inherit laya;
  laya-serve = writeShellScriptBin "laya-serve" ''
    exec ${pyEnv}/bin/python -m laya.serve "$@"
  '';
}
