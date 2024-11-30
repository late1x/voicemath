from voicemath.hme_parser.grammar import lex as lex
from voicemath import helpers
from voicemath.helpers.exceptions import LexicalError
from voicemath.hme_parser import correct_grammar as correct_grammar
from voicemath.hme_parser.grammar import lex as lex


class CheckLex():

    def __init__(self):
        self.__first_error = True
        self.latex_string = ""
        self.latex = ""
        self.latex_list = ""
        self.attempts = 0
        self.lex_error_list = None
        self.index = 0
        self.lex_errors_history = []
        self.pure_lex_errors = []

    def __locate_lex_error(self, lex_error_list):
        helpers.debug("\n[check_grammar_lex.py] __locate_lex_error()")

        error_list = lex_error_list.copy()
        latex = self.latex.copy()

        lex_errors = []
        lex_errors_history = self.lex_errors_history.copy()

        for error in error_list:

            if isinstance(error, dict):
                error = (error['label'], error['pos'])

            if error[1] != -1:

                count = 0
                count_list = 0

                latex_error_pos = error[1]  # position of char at string
                latex_error_token = error[0]  # It gets the token of the error

                # Search token (in latex) with position returned in error
                for symbol in latex:

                    if symbol['label'] != latex_error_token:

                        '''
                            Increment counter if current position is
                            the position containing the error
                        '''
                        count += len(symbol['label'])
                        count_list += 1

                    elif count == latex_error_pos:

                        lex_errors.append({'pos': latex_error_pos, 'pos_list': count_list, 'label': symbol['label'],
                            'prediction': symbol['prediction'], # It adds itself as a attempt of solution
                            'attempts': [symbol['label']]})

                        lex_errors_history.extend(lex_errors)

                        break

            else:
                helpers.debug("[check_grammar_lex.py] __locate_lex_error() | \
                    Cannot fix the error. \
                    Use an automata do fix it. \
                    Logging the error: {0}".format(error))
                continue

        return lex_errors, lex_errors_history

    def check_correct_lex(self):
        """Check and correct lex errors

        Args:
            latex_string (str): Latex string.
            latex (list): First latex structure.
            latex_list (list): [description]

        Returns:
            {
                'latex': self.latex,
                'latex_list': self.latex_list,
                'latex_string': self.latex_string,
                'lex_errors_history': self.lex_errors_history,
                'lex_pure_errors': self.pure_lex_errors
            }
        """

        second_lex_error_list = None
        lex_errors = []

        if not self.lex_error_list and self.__first_error and self.attempts < 3 and self.latex_string:

            lex_error_list = lex.LatexLexer(self.latex_string)

            if lex_error_list:
                self.pure_lex_errors.extend(lex_error_list)

                lex_errors, lex_errors_history = self.__locate_lex_error(lex_error_list)

                self.lex_error_list = lex_errors
                self.lex_errors_history = lex_errors_history
                self.__first_error = False

                self.__attempt_to_fix_error(lex_errors)

        elif self.lex_error_list and not self.__first_error and self.attempts < 3 and self.latex_string:

            second_lex_error_list = lex.LatexLexer(self.latex_string)

            if second_lex_error_list:

                self.pure_lex_errors.extend(second_lex_error_list)

                ''' HUM... CONFERIR.
                    If new error is EOF error
                    Remove the error from the list. Takes the next one.
                '''
                if second_lex_error_list[0][1] == -1:
                    second_lex_error_list.reverse()
                    second_lex_error_list.pop()
                    second_lex_error_list.reverse()

                lex_errors, lex_errors_history = self.__locate_lex_error(second_lex_error_list)

                self.lex_error_list = lex_errors
                self.lex_errors_history = lex_errors_history

                self.__attempt_to_fix_error(lex_errors)

        elif (self.lex_error_list and self.attempts >= 3) or not self.latex_string:
            raise LexicalError({'latex': self.latex, 'latex_list': self.latex_list, 'latex_string': self.latex_string,
                'error': self.lex_error_list,  # Current error
                'errors_history': self.lex_errors_history, 'pure_errors': self.pure_lex_errors})

        return {'latex': self.latex, 'latex_list': self.latex_list, 'latex_string': self.latex_string,
            'errors_history': self.lex_errors_history, 'pure_errors': self.pure_lex_errors}

    def __attempt_to_fix_error(self, lex_errors):

        # It tries to solve the FIRST error and returns an updated list of errors
        bg = correct_grammar.CorrectGrammar()

        # lex_errors: current error, self.lex_errors_history: all errors
        corrected_data = bg.correct_grammar_lex(lex_errors, self.latex, self.latex_list, 0, self.lex_errors_history)

        update_latex_string = corrected_data['latex_string']
        # Updated error with attempt
        self.lex_error_list = corrected_data['errors']
        self.index = corrected_data['index']

        # If there are remaining errors
        if self.lex_error_list:
            self.lex_errors_history = self.lex_error_list.copy()

        self.latex_string = update_latex_string
        self.attempts += 1
        return self.check_correct_lex()
