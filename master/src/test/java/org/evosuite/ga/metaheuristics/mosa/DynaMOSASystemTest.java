package org.evosuite.ga.metaheuristics.mosa;

import com.examples.with.different.packagename.BMICalculator;
import org.evosuite.EvoSuite;
import org.evosuite.Properties;
import org.evosuite.SystemTestBase;
import org.evosuite.ga.metaheuristics.GeneticAlgorithm;
import org.evosuite.strategy.TestGenerationStrategy;
import org.evosuite.testsuite.TestSuiteChromosome;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DynaMOSASystemTest extends SystemTestBase {
    private static final Logger logger = LoggerFactory.getLogger(DynaMOSASystemTest.class);

    @Before
    public void init() {
        //Properties.ENABLE_ADAPTIVE_PARAMETER_CONTROL = true;

        Properties.ENABLE_FITNESS_HISTORY = true;
        Properties.ENABLE_LANDSCAPE_ANALYSIS = true;
        //Properties.FITNESS_HISTORY_CAPACITY = 10;

        Properties.CLIENT_ON_THREAD = true;
        Properties.ASSERTIONS = false;

        //Properties.CRITERION = new Properties.Criterion[]{BRANCH, LINE};
        Properties.POPULATION = 1;
        Properties.SELECTION_FUNCTION = Properties.SelectionFunction.RANK_CROWD_DISTANCE_TOURNAMENT;

        Properties.NEW_STATISTICS = true;
        Properties.STATISTICS_BACKEND = Properties.StatisticsBackend.CSV;
        Properties.OUTPUT_VARIABLES = "Algorithm,TARGET_CLASS,Generations,criterion,Coverage,BranchCoverage,Total_Goals,Covered_Goals,NeutralityVolume,InformationContent";

        /*

        Properties.VIRTUAL_FS = true;
        Properties.VIRTUAL_NET = true;
        Properties.LOCAL_SEARCH_PROBABILITY = 1.0;
        Properties.LOCAL_SEARCH_RATE = 1;
        Properties.LOCAL_SEARCH_BUDGET_TYPE = Properties.LocalSearchBudgetType.TESTS;
        Properties.LOCAL_SEARCH_BUDGET = 100;
        Properties.SEARCH_BUDGET = 50000;

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

        Object result = evosuite.parseCommandLine(command);
        GeneticAlgorithm<?> ga = this.getGAFromResult(result);
        TestSuiteChromosome best = (TestSuiteChromosome) ga.getBestIndividual();

        logger.info("Best fitness: {}", best.getFitness());
        logger.info("EvolvedTestSuite: \n{}", best);

        int goals = TestGenerationStrategy.getFitnessFactories().get(0).getCoverageGoals().size();

        Assert.assertEquals("Wrong number of goals: ", 9, goals);
        Assert.assertEquals("Non-optimal coverage: ", 1d, best.getCoverage(), 0.001);
    }
}