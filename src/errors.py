class RAGError(Exception):
    """
    An expected error (bad input, missing index, ...).

    Its message is meant for the user: __main__ prints it without a
    traceback and exits with code 1.
    """
