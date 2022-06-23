package org.evosuite.landscape;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.LinkedHashMap;

/**
 * Represents the fitness history of the search.
 *
 * @author Kevin Haack
 */
public class FitnessHistory implements Serializable {
    /**
     * The logger.
     */
    private static final Logger logger = LoggerFactory.getLogger(FitnessHistory.class);
    /**
     * The fitness history, mapped by the age.
     */
    private final LinkedHashMap<Integer, Double> fitnessHistory = new LinkedHashMap<>();
    /**
     * The observed maximum fitness value.
     */
    private Double observedMaximum;
    /**
     * The observed minimum fitness value.
     */
    private Double observedMinimum;

    /**
     * Adds the passed fitness value to the history.
     *
     * @param age          The age of the fitness value.
     * @param fitnessValueRaw The fitness value.
     */
    public void addFitness(int age, double fitnessValueRaw) {
        if (fitnessValueRaw != Double.POSITIVE_INFINITY && fitnessValueRaw != Double.NEGATIVE_INFINITY) {
            // do not save infinity values, otherwise the epsilon calculation does not work
            if (null == this.observedMinimum || fitnessValueRaw < this.observedMinimum) {
                this.observedMinimum = fitnessValueRaw;
                logger.info("{} -> {} min", age, fitnessValueRaw);
            }

            if (null == this.observedMaximum || fitnessValueRaw > this.observedMaximum) {
                this.observedMaximum = fitnessValueRaw;
                logger.info("{} -> {} max", age, fitnessValueRaw);
            }

            this.fitnessHistory.put(age, fitnessValueRaw);
        }
    }

    /**
     * Returns the fitness history.
     *
     * @return Returns the fitness history.
     */
    public LinkedHashMap<Integer, Double> getFitnessHistory() {
        return fitnessHistory;
    }

    /**
     * Returns the observed maximum fitness value.
     *
     * @return Returns the observed maximum fitness value.
     */
    public Double getObservedMaximum() {
        return observedMaximum;
    }

    /**
     * Returns the observed minimum fitness value.
     *
     * @return Returns the observed minimum fitness value.
     */
    public Double getObservedMinimum() {
        return observedMinimum;
    }
}
