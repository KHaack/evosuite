/*
 * Copyright (C) 2010-2018 Gordon Fraser, Andrea Arcuri and EvoSuite
 * contributors
 *
 * This file is part of EvoSuite.
 *
 * EvoSuite is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3.0 of the License, or
 * (at your option) any later version.
 *
 * EvoSuite is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with EvoSuite. If not, see <http://www.gnu.org/licenses/>.
 */
package org.evosuite.maven.util;

import org.apache.maven.artifact.DependencyResolutionRequiredException;
import org.apache.maven.project.MavenProject;
import org.evosuite.ClientProcess;
import org.evosuite.PackageInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * @author José Campos
 */
public class ProjectUtils {
    private static final Logger logger = LoggerFactory.getLogger(ClientProcess.class);

    /**
     * Get compile elements (i.e., classes under /target/classes)
     *
     * @param project
     * @return
     */
    public static List<String> getCompileClasspathElements(MavenProject project) {
        List<String> compileClassPath = new ArrayList<>();

        try {
            project.getCompileClasspathElements()
                    .stream()
                    // we only target what has been compiled to a folder
                    .filter(element -> !element.endsWith(".jar"))
                    .filter(element -> new File(element).exists())
                    .forEach(element -> compileClassPath.add(element));
        } catch (DependencyResolutionRequiredException e) {
            logger.error("getCompileClasspathElements", e);
        }

        return compileClassPath;
    }

    /**
     * Get JUnit elements (i.e., classes under /target/test-classes) and compiled
     * elements (i.e., classes under /target/classes)
     *
     * @param project
     * @return
     */
    public static List<String> getTestClasspathElements(MavenProject project) {
        List<String> testClassPath = new ArrayList<>();

        try {
            project.getTestClasspathElements()
                    .stream()
                    // we only target what has been compiled to a folder
                    .filter(element -> !element.endsWith(".jar"))
                    .filter(element -> new File(element).exists())
                    .forEach(element -> testClassPath.add(element));
        } catch (DependencyResolutionRequiredException e) {
            logger.error("getTestClasspathElements", e);
        }

        return testClassPath;
    }

    /**
     * Get runtime elements
     *
     * @param project
     * @return
     */
    public static List<String> getRuntimeClasspathElements(MavenProject project) {
        List<String> runtimeClassPath = new ArrayList<>();

        try {
            project.getRuntimeClasspathElements()
                    .stream()
                    .filter(element -> new File(element).exists())
                    .forEach(element -> runtimeClassPath.add(element));
        } catch (DependencyResolutionRequiredException e) {
            logger.error("getRuntimeClasspathElements", e);
        }

        return runtimeClassPath;
    }

    /**
     * Get project's dependencies
     *
     * @param project
     * @return
     */
    public static List<String> getDependencyPathElements(MavenProject project) {
        List<String> dependencyArtifacts = new ArrayList<>();

        project.getDependencyArtifacts()
                .stream()
                .filter(element -> !element.isOptional())
                // FIXME do we really need to check the 'scope'?
                //.filter(element -> element.getScope().equals(scope))
                .filter(element -> element.getFile().exists())
                .filter(element -> !element.getGroupId().equals(PackageInfo.getEvoSuitePackage()))
                .filter(element -> !element.getGroupId().equals("junit"))
                .forEach(element -> dependencyArtifacts.add(element.getFile().getAbsolutePath()));

        return dependencyArtifacts;
    }

    /**
     * Convert a list of strings to a single string separated
     * by File.pathSeparator (i.e., ':')
     *
     * @param elements
     * @return
     */
    public static String toClasspathString(Collection<String> elements) {
        final StringBuilder str = new StringBuilder();

        elements.forEach(element ->
                str.append(str.length() == 0 ? element : File.pathSeparator + element)
        );

        return str.toString();
    }
}
