// `laya-java` lives in the laya monorepo so `scripts/gen_fixtures.py` can run the real Python in
// the same checkout. That is what makes the parity gate possible without a submodule or a
// published artifact to measure against.
rootProject.name = "laya-java-root"

// `laya-java-client` is deliberately NOT included. The directory holds no source -- the README
// lists the HTTP client as not implemented -- and including it gave it a full MavenPublication
// that `publishAllPublicationsToCentralRepository` would have pushed to Central as a zero-class
// jar whose POM promises an HTTP client. A version burned on Central cannot be replaced, so the
// module joins the build when it has code.
include("laya-java")
