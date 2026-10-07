package com.convaiinnovations.laya;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Guards the JDK matrix against being a lane that proves nothing.
 *
 * <p>The build's toolchain pins the COMPILER to 17, so a CI job that installs another JDK still
 * compiles and tests on 17 unless the test task's launcher is moved as well. When it is moved,
 * nothing downstream can confirm it happened: Gradle writes {@code <properties/>} empty into the
 * JUnit XML, so the report cannot tell 17 from 24 and neither can a later step reading it. The
 * running JVM is the only place that answer exists, so the assertion belongs here.
 *
 * <p>Not conditional on a checkpoint -- it is the JDK-sensitive half of this port that needs no
 * model. The character tables, the White_Space predicate, the pre-tokenizer's letter and number
 * classes and CPython's float {@code repr} are all compiled in from the reference precisely so
 * that the host JDK's own Unicode version cannot move them.
 */
final class TestJvmVersionTest {

    @Test
    @EnabledIfSystemProperty(named = "laya.test.expectedJavaVersion", matches = "\\d+")
    @DisplayName("the tests are running on the JDK the lane asked for")
    void runsOnTheRequestedJdk() {
        int expected = Integer.parseInt(System.getProperty("laya.test.expectedJavaVersion"));
        assertEquals(expected, Runtime.version().feature(),
                () -> "-PtestJavaVersion asked for JDK " + expected + " but the tests are running "
                      + "on " + Runtime.version()
                      + ". A matrix cell that silently falls back to the compile JDK tests the "
                      + "same thing twice and reports it as two passes.");
    }
}
