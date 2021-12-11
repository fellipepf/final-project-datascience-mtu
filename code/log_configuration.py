import logging

# create a logger object instance
logger = logging.getLogger()

# specifies the lowest severity for logging
logger.setLevel(logging.INFO)

# set a destination for your logs or a "handler"
# here, we choose to print on console (a consoler handler)
console_handler = logging.StreamHandler()

# set the logging format for your handler
log_format = '%(asctime)s | %(levelname)s: %(message)s'
console_handler.setFormatter(logging.Formatter(log_format))

# finally, we add the handler to the logger
logger.addHandler(console_handler)




if __name__ == '__main__':
    # start logging and show messages
    logger.debug('Here you have some information for debugging.')
    logger.info('Everything is normal. Relax!')
    logger.warning('Something unexpected but not important happend.')
    logger.error('Something unexpected and important happened.')
    logger.critical('OMG!!! A critical error happend and the code cannot run!')
