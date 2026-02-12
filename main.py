from llm_lab.utils.logger import setup_logger, get_logger

def main():
    setup_logger(name="llmtritonstack", auto_log_file=True)
    logger = get_logger("llmtritonstack")
    logger.info("Hello from llmtritonstack!")


if __name__ == "__main__":
    main()
