# { pkgs ? import <nixpkgs> {} , jts ? import <jts> {}}:

# with pkgs;
# let 
#   mavenRepository = jts.mvn2nix.buildMavenRepositoryFromLockFile { file = ./mvn2nix-lock.json; };
# in stdenv.mkDerivation rec {
#   pname = "tribuo";
#   version = "4.1.0";
#   name = "${pname}-${version}";

#   src = lib.cleanSource ./.;

#   nativeBuildInputs = [jdk maven];

#   LD_LIBRARY_PATH = jts.libgomp1 + "/lib";

#   buildPhase = ''
#     echo "Building with maven repository ${mavenRepository}"
#     mvn package --offline -Dmaven.repo.local=${mavenRepository}
#   '';

#   installPhase = ''
#     mkdir -p $out/lib
#     ln -s ${mavenRepository} $out/lib
    
#     ${findutils}/bin/find . -name '*.jar' -exec cp {} $out/ \;

#   '';
# }

{
  buildMavenRepositoryFromLockFile,
  findutils,
  lib,
  jdk,
  maven,
  libgomp1,
  stdenv
}:

let
  mavenRepository = buildMavenRepositoryFromLockFile { file = ./mvn2nix-lock.json; };
in stdenv.mkDerivation {
  pname = "tribuo";
  version = "4.3.0-SNAPSHOT";

  src = lib.cleanSource ./.;

  nativeBuildInputs = [jdk maven];

  LD_LIBRARY_PATH = libgomp1 + "/lib";

  buildPhase = ''
    mvn package --offline -Dmaven.repo.local=${mavenRepository}
  '';

  installPhase = ''
    mkdir -p $out/lib
    ln -s ${mavenRepository} $out/lib

    ${findutils}/bin/find . -name '*.jar' -exec cp {} $out/ \;
  '';
}
