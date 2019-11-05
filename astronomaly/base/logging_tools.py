import logging
from os import path


def setup_logger(log_filename="astronomaly.log"):
    root_logger = logging.getLogger()

    if len(root_logger.handlers) == 0:
        log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        root_logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(logging.WARNING)

        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    return root_logger


def format_function_call(func_name, *args, **kwargs):
    out_str = func_name + '('

    if len(args) != 0:
        for a in args:
            out_str += (str)(a) + ', '
    if len(kwargs.keys()) != 0:
        for k in kwargs.keys():
            out_str += ((str)(k) + '=' + (str)(kwargs[k]) + ', ')
    if out_str[-2] == ',':
        out_str = out_str[:-2]
    out_str += ')'

    return out_str


def log(msg, level='INFO'):
    root_logger = logging.getLogger()
    if len(root_logger.handlers) == 0:
        setup_logger()

    if level == 'ERROR':
        root_logger.error(msg)
    elif level == 'WARNING':
        root_logger.warning(msg)
    elif level == 'DEBUG':
        root_logger.debug(msg)
    else:
        root_logger.info(msg)


def check_if_inputs_same(class_name, local_variables):
    hdlrs = logging.getLogger().handlers
    # Try to be somewhat generic allowing for other handlers but this will only return the filename of the
    # first FileHandler object it finds. This should be ok except for weird logging edge cases.
    flname = ''
    checksum = 0
    for h in hdlrs:
        try:
            flname = h.baseFilename
            break
        except AttributeError:
            pass

    if len(flname) == 0 or not path.exists(flname):  # Log file doesn't exist yet
        return False

    else:
        fl = open(flname)
        func_args = {}
        args_same = False
        for ln in fl.readlines()[::-1]:
            if class_name + '(' in ln:
                s = ln.split('-')[-2].split(')')[0].split('(')[-1].split(',')
                if len(s) != 0:
                    for substring in s:
                        try:
                            key, value = substring.split('=')
                            func_args[key.strip()] = value.strip()
                        except ValueError:
                            # This happens when there are no arguments to the function
                            pass
                checksum_ln = ln.split('checksum:')
                if len(checksum_ln) > 1:
                    checksum = int(checksum_ln[-1])
                else:
                    checksum = 0
                args_same = True

                for k in func_args.keys():
                    if k not in local_variables.keys():
                        args_same = False
                        break
                    else:
                        if func_args[k] != (str)(local_variables[k]):
                            args_same = False
                            break
                break

        return args_same, checksum
