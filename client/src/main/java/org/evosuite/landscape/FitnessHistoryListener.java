package org.evosuite.landscape;

import org.evosuite.Properties;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.metaheuristics.GeneticAlgorithm;
import org.evosuite.ga.metaheuristics.SearchListener;
import org.evosuite.rmi.ClientServices;
import org.evosuite.statistics.RuntimeVariable;
import org.evosuite.testcase.execution.ExecutionTraceImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The {@link SearchListener} that keep track of the fitness history.
 *
 * @param <T> The chromosome type.
 * @author Kevin Haack
 */
public class FitnessHistoryListener<T extends Chromosome<T>> implements SearchListener<T> {
    /**
     * The percentage steps of the collections.
     */
    private static final double COLLECTION_STEPS = 0.1;

    /**
     * The logger.
     */
    private static final Logger logger = LoggerFactory.getLogger(FitnessHistoryListener.class);
    /**
     * The fitness history.
     */
    private final FitnessHistory fitnessHistory = new FitnessHistory();
    /**
     * The start time of the algorithm.
     */
    private long startTime;
    /**
     * List of all collected neutrality volumes.
     */
    private List<Integer> neutralityVolumes;
    /**
     * List of all collected information contents.
     */
    private List<Double> informationContents;
    private List<Double> fitnessRatio;
    private List<Integer> generations;
    private List<Integer> gradientBranches;
    private List<Integer> gradientBranchesCovered;
    /**
     * The percentage of the next collection.
     */
    private double percentageNextCollection;

    @Override
    public void searchStarted(GeneticAlgorithm<T> algorithm) {
        logger.info("searchStarted");

        neutralityVolumes = new LinkedList<>();
        informationContents = new LinkedList<>();
        fitnessRatio = new LinkedList<>();
        generations = new LinkedList<>();
        gradientBranchesCovered = new LinkedList<>();
        gradientBranches = new LinkedList<>();

        startTime = System.currentTimeMillis();
        percentageNextCollection = COLLECTION_STEPS;

        ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable._ParameterControlled, "");
    }

    @Override
    public void iteration(GeneticAlgorithm<T> algorithm) {
        if (Properties.ENABLE_LANDSCAPE_ANALYSIS) {
            trackValues(algorithm, false);
        }

        if (Properties.ENABLE_PARAMETER_CONTROL) {
            if (isPercentageReached(Properties.PC_AT)) {
                ParameterControl parameterControl = new ParameterControl(generations, neutralityVolumes, informationContents, fitnessRatio, gradientBranchesCovered, gradientBranches);
                boolean shouldControl;

                switch (Properties.PC_Prediction) {
                    case HIGH_STDEV:
                        shouldControl =!parameterControl.performsWell();
                        throw new RuntimeException("Not supported... change to one tree. and inverse");
                    case LOW_END_COVERAGE:
                        shouldControl = !parameterControl.lowCoverageStd();
                        throw new RuntimeException("Not supported... change to one tree. and inverse");
                    case RELATIVE_LOW_COVERAGE:
                        shouldControl = !parameterControl.relativeHighCoverage();
                        throw new RuntimeException("Not supported... change to one tree. and inverse");
                    case HIGH_STDEV_LOW_END_COVERAGE:
                        shouldControl = !parameterControl.lowCoverageStd() && !parameterControl.performsWell();
                        throw new RuntimeException("Not supported... change to one tree. and inverse");
                    case HIGH_STDEV_RELATIVE_LOW_COVERAGE:
                        shouldControl = parameterControl.highStDevRelativeLowCoverage();
                        break;
                    case HIGHER_WITH_POP125_and_RELATIVE_LOW_COVERAGE:
                        shouldControl = parameterControl.pop125RelativeLowCoverage();
                        break;
                    case ALWAYS:
                        shouldControl = true;
                        break;
                    case NONE:
                    default:
                        shouldControl = false;
                }

                if (shouldControl) {
                    ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable._ParameterControlled, "yes");
                    Properties.CROSSOVER_RATE = Properties.PC_CROSSOVER_RATE;
                    Properties.POPULATION = Properties.PC_POPULATION;
                    Properties.NUMBER_OF_MUTATIONS = Properties.PC_NUMBER_OF_MUTATIONS;
                    Properties.RANK_BIAS = Properties.PC_RANK_BIAS;
                    Properties.P_TEST_INSERTION = Properties.PC_P_TEST_INSERTION;
                } else {
                    ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable._ParameterControlled, "no");
                }
            }
        }
    }

    /**
     * Returns how many percentage of the search budged are reached.
     *
     * @return Returns how many percentage of the search budget are reached.
     */
    private double getReachedPercentage() {
        long currentTime = System.currentTimeMillis();
        return (double) (currentTime - startTime) / ((double) Properties.SEARCH_BUDGET * 1000);
    }

    /**
     * Indicated whether the passed percentage of the search budget was reached.
     *
     * @return True if reached.
     */
    private boolean isPercentageReached(double percentage) {
        return getReachedPercentage() >= percentage;
    }

    /**
     * Track all values.
     *
     * @param algorithm The algorithm.
     */
    private void trackValues(GeneticAlgorithm<T> algorithm, boolean forceSave) {
        // fitness
        double fitnessRaw = 0;

        for (T c : algorithm.getPopulation()) {
            fitnessRaw += c.getFitness();
        }
        this.fitnessHistory.addFitness(algorithm.getAge(), fitnessRaw);

        // on every percentage step
        if (forceSave || isPercentageReached(percentageNextCollection)) {
            NeutralityVolume nv = new NeutralityVolume(this.fitnessHistory);
            nv.init();

            // generations
            generations.add(algorithm.getAge());

            // neutralityVolume
            neutralityVolumes.add(nv.getNeutralityVolume());

            // informationContent
            informationContents.add(nv.getInformationContent());

            // fitnessRatio
            double fitness = 1.0;
            if (null != this.fitnessHistory.getObservedMaximum() && fitnessRaw < this.fitnessHistory.getObservedMaximum()) {
                fitness = fitnessRaw / this.fitnessHistory.getObservedMaximum();
            }
            fitnessRatio.add(fitness);

            /*
             * branches
             */
            //int numBranches = BranchPool.getInstance(TestGenerationContext.getInstance().getClassLoaderForSUT()).getBranchCounter() * 2;

            if (Properties.TRACK_BOOLEAN_BRANCHES) {
                int gradientBranchCount = ExecutionTraceImpl.gradientBranches.size() * 2;
                this.gradientBranches.add(gradientBranchCount);
            }

            if (Properties.TRACK_COVERED_GRADIENT_BRANCHES) {
                int coveredGradientBranchCount = ExecutionTraceImpl.gradientBranchesCoveredTrue.size()
                        + ExecutionTraceImpl.gradientBranchesCoveredFalse.size();
                this.gradientBranchesCovered.add(coveredGradientBranchCount);
            }

            /*Set<Integer> coveredTrueBranches = new HashSet<>();
            Set<Integer> coveredFalseBranches = new HashSet<>();
            Set<String> coveredBranchlessMethods = new HashSet<>();
            for (T c : algorithm.getPopulation()) {
                if (c instanceof TestChromosome) {
                    TestChromosome test = (TestChromosome) c;
                    ExecutionTrace trace = test.getLastExecutionResult().getTrace();

                    coveredTrueBranches.addAll(trace.getCoveredTrueBranches());
                    coveredFalseBranches.addAll(trace.getCoveredFalseBranches());
                    coveredBranchlessMethods.addAll(trace.getCoveredBranchlessMethods());
                }
            }*/
            // covered
            // coveredTrueBranches.size() + coveredFalseBranches.size()
            // branchless
            // coveredBranchlessMethods.size()

            // next step at
            percentageNextCollection += COLLECTION_STEPS;
        }
    }

    @Override
    public void searchFinished(GeneticAlgorithm<T> algorithm) {
        logger.info("searchFinished");

        if (Properties.ENABLE_LANDSCAPE_ANALYSIS) {
            trackValues(algorithm, true);

            // fill statistic
            ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable._FitnessMin, this.fitnessHistory.getObservedMinimum());
            ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable._FitnessMax, this.fitnessHistory.getObservedMaximum());

            String sequence = "[" + neutralityVolumes.stream().map(String::valueOf).collect(Collectors.joining(";")) + "]";
            ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable._NeutralityVolume, sequence);

            sequence = "[" + informationContents.stream().map(String::valueOf).collect(Collectors.joining(";")) + "]";
            ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable._InformationContent, sequence);

            sequence = "[" + fitnessRatio.stream().map(String::valueOf).collect(Collectors.joining(";")) + "]";
            ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable._FitnessRatio, sequence);

            sequence = "[" + generations.stream().map(String::valueOf).collect(Collectors.joining(";")) + "]";
            ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable._Generations, sequence);

            if (Properties.TRACK_BOOLEAN_BRANCHES) {
                sequence = "[" + gradientBranches.stream().map(String::valueOf).collect(Collectors.joining(";")) + "]";
                ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable._GradientBranches, sequence);
            }
            if (Properties.TRACK_COVERED_GRADIENT_BRANCHES) {
                sequence = "[" + gradientBranchesCovered.stream().map(String::valueOf).collect(Collectors.joining(";")) + "]";
                ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable._GradientBranchesCovered, sequence);
            }

            // /////////////////////////
            NeutralityVolume nv = new NeutralityVolume(this.fitnessHistory);
            nv.init();
            nv.printHistory();
        }
    }

    @Override
    public void fitnessEvaluation(T individual) {
    }

    @Override
    public void modification(T individual) {
    }
}
