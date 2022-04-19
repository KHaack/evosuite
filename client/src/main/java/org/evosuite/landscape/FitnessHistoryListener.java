package org.evosuite.landscape;

import org.evosuite.ga.Chromosome;
import org.evosuite.ga.metaheuristics.GeneticAlgorithm;
import org.evosuite.ga.metaheuristics.SearchListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The {@link SearchListener} that keep track of the fitness history.
 *
 * @param <T> The chromosome type.
 * @author Kevin Haack
 */
public class FitnessHistoryListener<T extends Chromosome<T>> implements SearchListener<T> {
    /**
     * The logger.
     */
    private static final Logger logger = LoggerFactory.getLogger(FitnessHistoryListener.class);
    /**
     * The fitness history.
     */
    private final FitnessHistory fitnessHistory = new FitnessHistory();

    @Override
    public void searchStarted(GeneticAlgorithm<T> algorithm) {
        logger.info("searchStarted");
    }

    @Override
    public void iteration(GeneticAlgorithm<T> algorithm) {
        this.fitnessHistory.addFitness(algorithm.getAge(), algorithm.getBestIndividual().getFitness());
    }

    @Override
    public void searchFinished(GeneticAlgorithm<T> algorithm) {
        logger.info("searchFinished");
        this.fitnessHistory.printHistory();
    }

    @Override
    public void fitnessEvaluation(T individual) {
    }

    @Override
    public void modification(T individual) {
    }
}
