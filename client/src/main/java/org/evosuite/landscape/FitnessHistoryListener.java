package org.evosuite.landscape;

import org.evosuite.Properties;
import org.evosuite.ga.Chromosome;
import org.evosuite.ga.metaheuristics.GeneticAlgorithm;
import org.evosuite.ga.metaheuristics.SearchListener;
import org.evosuite.rmi.ClientServices;
import org.evosuite.statistics.RuntimeVariable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

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
        double fitness = 0;

        for(T c: algorithm.getPopulation()) {
            fitness += c.getFitness();
        }

        this.fitnessHistory.addFitness(algorithm.getAge(), fitness);
    }

    @Override
    public void searchFinished(GeneticAlgorithm<T> algorithm) {
        logger.info("searchFinished");

        if(Properties.ENABLE_LANDSCAPE_ANALYSIS) {
            NeutralityVolume nv = new NeutralityVolume(this.fitnessHistory);
            nv.init();

            nv.printHistory();

            ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable.FitnessMax, nv.getFitnessHistory().getObservedMaximum());
            ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable.FitnessMin, nv.getFitnessHistory().getObservedMinimum());
            ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable.NeutralityVolume, nv.getNeutralityVolume());
            ClientServices.getInstance().getClientNode().trackOutputVariable(RuntimeVariable.InformationContent, nv.getInformationContent());
        }
    }

    @Override
    public void fitnessEvaluation(T individual) {
    }

    @Override
    public void modification(T individual) {
    }
}
