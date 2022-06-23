package org.evosuite.ga.metaheuristics.mosa;

import com.examples.with.different.packagename.BMICalculator;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;
import org.evosuite.EvoSuite;
import org.evosuite.Properties;
import org.evosuite.SystemTestBase;
import org.evosuite.ga.metaheuristics.GeneticAlgorithm;
import org.evosuite.result.TestGenerationResult;
import org.evosuite.strategy.TestGenerationStrategy;
import org.evosuite.testsuite.TestSuiteChromosome;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class DynaMOSASystemTest extends SystemTestBase {
    private static final Logger logger = LoggerFactory.getLogger(DynaMOSASystemTest.class);

    @Before
    public void init() {
        //Properties.ENABLE_ADAPTIVE_PARAMETER_CONTROL = true;

        Properties.ENABLE_FITNESS_HISTORY = true;
        Properties.ENABLE_LANDSCAPE_ANALYSIS = true;
        Properties.ENABLE_PARAMETER_CONTROL = true;

        Properties.CLIENT_ON_THREAD = true;
        Properties.ASSERTIONS = false;

        //Properties.CRITERION = new Properties.Criterion[]{BRANCH, LINE};
        Properties.POPULATION = 1;

        Properties.NEW_STATISTICS = true;
        Properties.STATISTICS_BACKEND = Properties.StatisticsBackend.CSV;

        Properties.TRACK_COVERED_GRADIENT_BRANCHES = true;
        Properties.TRACK_BOOLEAN_BRANCHES = true;

        //Properties.SEARCH_BUDGET = 2000;

        /*

        Properties.VIRTUAL_FS = true;
        Properties.VIRTUAL_NET = true;
        Properties.LOCAL_SEARCH_PROBABILITY = 1.0;
        Properties.LOCAL_SEARCH_RATE = 1;
        Properties.LOCAL_SEARCH_BUDGET_TYPE = Properties.LocalSearchBudgetType.TESTS;
        Properties.LOCAL_SEARCH_BUDGET = 100;


        Properties.RESET_STATIC_FIELD_GETS = true;

        Properties.STOPPING_CONDITION = Properties.StoppingCondition.MAXTIME;
        Properties.SEARCH_BUDGET = 60;
        Properties.MINIMIZATION_TIMEOUT = 60;
        Properties.ASSERTION_TIMEOUT = 60;

        Properties.CRITERION = new Properties.Criterion[]{Properties.Criterion.BRANCH};

        Properties.MINIMIZE = true;
        Properties.ASSERTIONS = true;*/
    }

    @Test
    public void simpleTest() {
        EvoSuite evosuite = new EvoSuite();

        //String targetClass = NullString.class.getCanonicalName();
        //String targetClass = TargetMethod.class.getCanonicalName();
        String targetClass = BMICalculator.class.getCanonicalName();

        String[] command = new String[]{"-generateMOSuite", "-class", targetClass};
        Properties.ALGORITHM = Properties.Algorithm.DYNAMOSA;
        Properties.SELECTION_FUNCTION = Properties.SelectionFunction.RANK_CROWD_DISTANCE_TOURNAMENT;
        Properties.OUTPUT_VARIABLES = "Algorithm,TARGET_CLASS,Generations,criterion,Coverage,BranchCoverage,Total_Goals,Covered_Goals,Fitness,_FitnessMax,_FitnessMin,_NeutralityVolume,_InformationContent,_FitnessRatio,_Generations,_GradientBranches,_GradientBranchesCovered";

        Object result = evosuite.parseCommandLine(command);
        GeneticAlgorithm<?> ga = this.getGAFromResult(result);
        TestSuiteChromosome best = (TestSuiteChromosome) ga.getBestIndividual();

        logger.info("Best fitness: {}", best.getFitness());
        logger.info("EvolvedTestSuite: \n{}", best);

        int goals = TestGenerationStrategy.getFitnessFactories().get(0).getCoverageGoals().size();

        Assert.assertEquals("Wrong number of goals: ", 9, goals);
        Assert.assertEquals("Non-optimal coverage: ", 1d, best.getCoverage(), 0.001);
    }

    @Test
    public void randomTest() throws IOException, CsvException {
        EvoSuite evosuite = new EvoSuite();

        //String targetClass = NullString.class.getCanonicalName();
        //String targetClass = TargetMethod.class.getCanonicalName();
        String targetClass = BMICalculator.class.getCanonicalName();

        String[] command = new String[]{"-generateRandom", "-class", targetClass};
        Properties.ALGORITHM = Properties.Algorithm.RANDOM_SEARCH;
        Properties.OUTPUT_VARIABLES = "Algorithm,TARGET_CLASS,Generations,criterion,Coverage,BranchCoverage,Total_Goals,Covered_Goals";

        Object result = evosuite.parseCommandLine(command);

        Assert.assertNotNull(result);

        String statistics_file = System.getProperty("user.dir") + File.separator + Properties.REPORT_DIR + File.separator + "statistics.csv";
        System.out.println("Statistics file " + statistics_file);

        CSVReader reader = new CSVReader(new FileReader(statistics_file));
        List<String[]> rows = reader.readAll();
        assertEquals(2, rows.size());
        reader.close();

        assertEquals("RANDOM_SEARCH", rows.get(1)[0]); // TARGET_CLASS
        assertEquals(targetClass, rows.get(1)[1]); // TARGET_CLASS
        // Generations
        assertEquals("BRANCH", rows.get(1)[3]); // criterion
        assertEquals("1.0", rows.get(1)[4]); // Coverage
        assertEquals("1.0", rows.get(1)[5]); // BranchCoverage
        assertEquals("9", rows.get(1)[6]); // Total_Goals
        assertEquals("9", rows.get(1)[7]); // Covered_Goals
    }
}