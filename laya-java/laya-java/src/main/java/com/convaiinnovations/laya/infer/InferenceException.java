package com.convaiinnovations.laya.infer;

/**
 * An inference backend failed.
 *
 * <p>Unchecked, and deliberately not ONNX Runtime's own exception type. A caller asking a question
 * about a document cannot do anything useful about an {@code OrtException} except log it, and
 * making every {@code predict} signature declare one would spread a backend's checked exception
 * across an API that is meant to outlive that backend.
 */
public final class InferenceException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public InferenceException(String message, Throwable cause) {
        super(message, cause);
    }

    public InferenceException(String message) {
        super(message);
    }
}
