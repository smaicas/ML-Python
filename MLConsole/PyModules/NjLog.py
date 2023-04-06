import colorama
import logging
import inspect

colorama.init()

logging.basicConfig(level=logging.INFO,
                    format=f"{colorama.Fore.WHITE}%(asctime)s{colorama.Style.RESET_ALL} %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def log(message):
    logger.info(f"{colorama.Fore.BLUE}INFO{colorama.Style.RESET_ALL} {message}")


def logwarn(message):
    logger.warning(f"{colorama.Fore.YELLOW}WARNING{colorama.Style.RESET_ALL} {message}")


def logerror(message):
    logger.error(f"{colorama.Fore.RED}ERROR{colorama.Style.RESET_ALL} {message}")


def logcall():
    caller = inspect.stack()[1].function
    logger.info(f"{colorama.Fore.CYAN}Call {caller}{colorama.Style.RESET_ALL}")


def logok(message=None):
    message = message if message else "OK"
    caller = inspect.stack()[1].function
    logger.info(f"{colorama.Fore.GREEN}{message} {colorama.Style.BRIGHT}\u2713 {caller}{colorama.Style.RESET_ALL}")
