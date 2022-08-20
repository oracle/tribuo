{
  jdk,
  python3,
  stdenv,
  fetchFromGitHub,
  lib
}:

let
  pname = "IJava";
  version = "1.3.0";
  src = fetchFromGitHub {
    owner = "SpencerPark";
    repo = pname;
    rev = "v${version}";
    sha256 = lib.fakeSha256;
  };
in stdenv.mkDerivation {
  inherit pname version src;

  buildPhase = ''
    ${python3}/bin/python $src/install.py -h
  '';
}

