import sys


#THIS CLASS ALOWS US TO EFFICIENTLY HANDLING EXEPTIONS

def get_error_message(error, error_details:sys):
    _,_, exc_info = error_details.exc_info()
    file_name = exc_info.tb_frame.f_code.co_filename
    error_message = f"Error occured in file {file_name}, in line number {exc_info.tb_lineno}. Error details: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = get_error_message(error_message, error_details)
    
    def __str__(self):
        return self.error_message
    


