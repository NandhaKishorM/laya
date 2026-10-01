# The laya python package + a laya-serve runner, built from whatever `pkgs` is
# in scope. Factored out so both the flake's packages/overlay AND the NixOS
# module can build it the same way -- the module cannot rely on an overlay,
# because hosts that inject `pkgs` via specialArgs ignore module-level
# `nixpkgs.overlays`.
{ lib, python3, python3Packages, writeShellScriptBin }:
let
  laya = python3Packages.buildPythonPackage {
    pname = "laya";
    # pyproject.toml is the source of truth for this string. Reading it here would
    # need an eval-time TOML import; instead the equality is enforced from the Python
    # side, where tests/test_packaging.py compares all three declarations (this one,
    # pyproject.toml and laya.__version__) and fails CI when they diverge. Nothing
    # did, which is how this sat at 0.3.4 while the package reached 0.3.20.
    version = "0.3.22";
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
