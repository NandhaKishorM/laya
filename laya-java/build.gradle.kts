plugins {
    `java-library`
    `maven-publish`
    signing
}

// JDK 17: the oldest LTS still receiving updates, and the first with records and sealed
// interfaces, which the answer types need -- a ChoiceAnswer is not a ScoreAnswer, and a port that
// models them as one map loses that at the API boundary.
val javaRelease = 17

subprojects {
    apply(plugin = "java-library")
    apply(plugin = "maven-publish")
    apply(plugin = "signing")

    group = "com.convaiinnovations"
    version = providers.gradleProperty("layaJavaVersion").getOrElse("0.1.0-SNAPSHOT")

    repositories { mavenCentral() }

    extensions.configure<JavaPluginExtension> {
        toolchain { languageVersion.set(JavaLanguageVersion.of(javaRelease)) }
        // Maven Central requires both; they are also what makes the runtime readable from an IDE,
        // which matters more than usual here because the comments carry the parity reasoning.
        withSourcesJar()
        withJavadocJar()
    }

    tasks.withType<JavaCompile>().configureEach {
        options.release.set(javaRelease)
        // -Xlint:all with -Werror: a port's silent narrowing conversion is exactly the class of
        // bug the fixtures cannot see, because it changes a number without changing a shape.
        options.compilerArgs.addAll(listOf("-Xlint:all", "-Werror"))
        options.encoding = "UTF-8"
    }

    tasks.withType<Javadoc>().configureEach {
        (options as StandardJavadocDocletOptions).apply {
            encoding = "UTF-8"
            charSet = "UTF-8"
            // Javadoc's HTML checker is stricter than the compiler and fails on things that are
            // not defects; the compiler's -Werror is the gate that matters.
            addStringOption("Xdoclint:none", "-quiet")
        }
    }

    // A jar that is byte-identical from the same source, so a consumer can verify what they got.
    tasks.withType<AbstractArchiveTask>().configureEach {
        isPreserveFileTimestamps = false
        isReproducibleFileOrder = true
    }

    tasks.withType<Test>().configureEach {
        useJUnitPlatform()
        testLogging { events("failed", "skipped") }
        // Run the tests on a DIFFERENT JDK from the one the classes are compiled for, when asked:
        //
        //     ./gradlew test -PtestJavaVersion=24
        //
        // `options.release` above keeps the bytecode at 17 whatever this is set to, so the
        // artifact a consumer gets does not change; only the JVM executing the tests does.
        //
        // This exists because the toolchain pins the compiler to 17, so a CI matrix that merely
        // installs another JDK still compiles AND tests on 17 -- a green lane proving nothing.
        // The property it buys is real and was bought the hard way: `\p{L}` in java.util.regex
        // follows the JDK's own Unicode version, so the pre-tokenizer returned different token
        // ids from the same jar on different JDKs. Corretto 17 carries Unicode 13.0 and Corretto
        // 24 carries 16.0, and `Character.isLetter` disagrees with itself across them on 751 of
        // the code points this port had to classify. The classes are compiled in from the
        // reference now, so the answer must not move -- and this is what runs that check.
        val testJavaVersion = providers.gradleProperty("testJavaVersion")
        if (testJavaVersion.isPresent) {
            val want = testJavaVersion.get().trim().toInt()
            val toolchains = project.extensions.getByType<JavaToolchainService>()
            // `launcherFor` fails the build when no such JDK is installed, provided
            // auto-download is off, so a lane asking for a JDK it does not have cannot quietly
            // fall back to 17 and report a pass for the wrong JVM.
            javaLauncher.set(toolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(want)) })
            // And the suite checks it from the inside, because the JUnit XML does not record
            // which JVM produced it: `<properties/>` comes out empty, so neither the report nor
            // a later step can tell 17 from 24. Asserting it in-process is the only place the
            // answer actually exists.
            systemProperty("laya.test.expectedJavaVersion", want.toString())
        }
        // Passed through rather than inherited silently, so a lane that forgets them is a lane
        // whose parity tests abort loudly instead of one that quietly tests less.
        //
        // Blank counts as absent. A CI matrix cell that does not own a graph writes
        // `LAYA_ONNX_GRAPH: ''` rather than leaving the variable out -- an expression yielding
        // the empty string still DEFINES the variable -- and an empty path handed to the tests
        // is a path that cannot be opened, so the graph-backed factories would fail where they
        // are supposed to abort by assumption. Treating blank as unset is what makes "this cell
        // has no graph" and "this machine has no graph" the same case, which is what the tests
        // are written against.
        listOf("LAYA_CHECKPOINTS", "LAYA_ONNX_GRAPH", "LAYA_PREDICT_GOLDEN").forEach { name ->
            System.getenv(name)?.takeIf { it.isNotBlank() }?.let { environment(name, it) }
        }
    }

    dependencies {
        "testImplementation"("org.junit.jupiter:junit-jupiter:5.11.4")
        "testRuntimeOnly"("org.junit.platform:junit-platform-launcher")
    }

    extensions.configure<PublishingExtension> {
        repositories {
            // The Central Portal's Maven-compatible staging endpoint, so publishing needs no
            // third-party Gradle plugin and therefore no extra build dependency to audit. A
            // maintainer who prefers the portal's own plugin can swap this block out; nothing else
            // in the build depends on it.
            //
            // NOT EXERCISED. Publishing needs a verified `com.convaiinnovations` namespace, a
            // Central token and a PGP key, all of which belong to the project owner, so this
            // configuration is written from Central's documented requirements and has never been
            // run. `publishToMavenLocal` is the part that is tested.
            maven {
                name = "Central"
                url = uri("https://ossrh-staging-api.central.sonatype.com/service/local/staging/deploy/maven2/")
                credentials {
                    username = providers.environmentVariable("MAVEN_CENTRAL_USERNAME").orNull
                    password = providers.environmentVariable("MAVEN_CENTRAL_PASSWORD").orNull
                }
            }
        }
        publications {
            create<MavenPublication>("maven") {
                from(components["java"])
                pom {
                    name.set(project.name)
                    description.set(
                        when (project.name) {
                            "laya-java" ->
                                "Laya on the JVM: a fast, local, non-autoregressive decision model " +
                                "runtime (tokenizer, sequence builder, ONNX inference, typed answers)."
                            else ->
                                "HTTP client for a self-hosted laya-serve, sharing laya-java's " +
                                "typed answers."
                        }
                    )
                    url.set("https://github.com/NandhaKishorM/laya")
                    licenses {
                        license {
                            name.set("Apache-2.0")
                            url.set("https://www.apache.org/licenses/LICENSE-2.0")
                        }
                    }
                    developers {
                        developer {
                            name.set("Convai Innovations")
                            url.set("https://github.com/NandhaKishorM")
                        }
                    }
                    scm {
                        url.set("https://github.com/NandhaKishorM/laya")
                        connection.set("scm:git:https://github.com/NandhaKishorM/laya.git")
                        developerConnection.set("scm:git:ssh://git@github.com/NandhaKishorM/laya.git")
                    }
                }
            }
        }
    }

    // Signing is required by Maven Central and impossible without the project's key, so it is
    // configured to activate only when one is supplied. `gradlew build` and
    // `publishToMavenLocal` therefore work for anyone; a release needs the maintainer.
    extensions.configure<SigningExtension> {
        val signingKey = providers.environmentVariable("LAYA_SIGNING_KEY").orNull
        val signingPassword = providers.environmentVariable("LAYA_SIGNING_PASSWORD").orNull
        isRequired = signingKey != null
        if (signingKey != null) {
            useInMemoryPgpKeys(signingKey, signingPassword)
            sign(extensions.getByType<PublishingExtension>().publications)
        }
    }
}
