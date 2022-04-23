package org.evosuite.landscape;

import java.io.Serializable;
import java.util.Objects;

public class SubBlock implements Serializable {
    private final int first;
    private final int second;

    public SubBlock(int first, int second) {
        this.first = first;
        this.second = second;
    }

    public int getFirst() {
        return first;
    }

    public int getSecond() {
        return second;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SubBlock subBlock = (SubBlock) o;
        return first == subBlock.first && second == subBlock.second;
    }

    @Override
    public int hashCode() {
        return first * 10 + second;
    }
}
