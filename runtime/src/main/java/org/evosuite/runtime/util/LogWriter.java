package org.evosuite.runtime.util;

import org.slf4j.Logger;
import org.slf4j.event.Level;

import java.io.IOException;
import java.io.Writer;
import java.util.Arrays;

/**
 * Prints to a text-output on the log-level. This class implements all methods found in Writer.
 *
 * @author Kevin Haack
 */
public class LogWriter extends Writer {

    /**
     * The logger to write to.
     */
    private final Logger logger;
    /**
     * The log level to write to.
     */
    private final Level level;

    public LogWriter(Logger logger, Level level) {
        this.logger = logger;
        this.level = level;
    }

    /**
     * Logs the passed message.
     *
     * @param message The message to be logged.
     */
    private void log(String message) {
        switch (this.level) {
            case DEBUG:
                this.logger.debug(message);
                break;
            case INFO:
                this.logger.info(message);
                break;
            case TRACE:
                this.logger.trace(message);
                break;
            case WARN:
                this.logger.warn(message);
                break;
            case ERROR:
            default:
                this.logger.error(message);
        }
    }

    @Override
    public void write(char[] cbuf, int off, int len) throws IOException {
        this.log(new String(Arrays.copyOfRange(cbuf, off, len)));
    }

    @Override
    public void write(int c) throws IOException {
        this.log(String.valueOf((char)c));
    }

    @Override
    public void write(char[] cbuf) throws IOException {
        this.log(new String(cbuf));
    }

    @Override
    public void write(String str) throws IOException {
        this.log(str);
    }

    @Override
    public void write(String str, int off, int len) throws IOException {
        this.log(str.substring(off, len));
    }

    @Override
    public void flush() throws IOException {

    }

    @Override
    public void close() throws IOException {

    }
}
