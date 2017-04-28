"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import sys
import timeit

import isolation

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
<<<<<<< HEAD
    return(heuristic_F(game, player))


def heuristic_A(game, player):
    """
    Worst Performance of the lot.
    If the Student has fewer move options than the opponent, the function returns - (# moves the opponent has)/(# moves the computer has)**2.
    If the Student has more move options than the opponent, the function returns (# moves the opponent has)/(# moves the computer has)**2.
    The logic for this is similar. If a move causes the Student to have greatly proportionally fewer moves than the opponent, it is a bad move.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    own_moves = (game.get_legal_moves(player))
    opp_moves = (game.get_legal_moves(game.get_opponent(player)))

    if len (own_moves) == 0 and len (opp_moves) != 0:         #If this move results in the opponent winning the game, return utility -infinity
        return float ('-inf')
    elif len (own_moves) != 0 and len (opp_moves) == 0:       #If this move results in the computer winning the game, return utility infinity
        return float ('inf')
    elif len (own_moves) == 0 and len (opp_moves) == 0:       #If this move results in a draw, return utility -10
        return -10
    elif len (own_moves) >= len (opp_moves):
        return (len (own_moves) / len (opp_moves))**2
    elif len (own_moves) < len (opp_moves):
        return - (len (opp_moves)/ len(own_moves))**2


def heuristic_B(game, player):
    """
    Aggressive play in the first half of the game. Active player will try to choose the most aggressive move.
    Heuristic calculates number of players move vs 3.5 of value of an opponent’s moves.
    In the second half of the game heuristic will calculate number of players move vs number of an opponent’s moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    cells_left = game.width * game.height - game.move_count

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if cells_left < int((game.width * game.height) / 2):
        return float(own_moves - opp_moves)
    return float(own_moves - 3 * opp_moves)


def heuristic_C(game, player):
    """Similar to "B", Most aggressive initially, then drop aggressiveness at 1/3 and further more at 1/4 moves remaining.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    board_size = game.width * game.height
    cells_left = board_size - game.move_count
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if cells_left < int((board_size) / 4):
        return float(own_moves - opp_moves)
    if cells_left < int((board_size) / 3):
        return float(own_moves - 2 * opp_moves)
    return float(own_moves - 3 * opp_moves)


def heuristic_D(game, player):
    """Similar to "B", Least aggressive initially, then increase aggressiveness at 1/3 and further more at 1/4 moves remaining.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    board_size = game.width * game.height
    cells_left = board_size - game.move_count
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if cells_left < int((board_size) / 4):
        return float(own_moves - 3 * opp_moves)
    if cells_left < int((board_size) / 3):
        return float(own_moves - 2 * opp_moves)
    return float(own_moves - opp_moves)

def heuristic_E(game, player):
    """Heuristic base don the following formula, (own_moves ^2) / opp moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    if (opp_moves == 0 ):
        return float( (own_moves ^ 2) )
    return float( (own_moves ^ 2) / opp_moves)

def heuristic_F(game, player):
    """Best Heuristic from the test results, Similar to "E" : (own_moves ^3) / opp moves

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    if (opp_moves == 0 ):
        return float( (own_moves ^ 3) )
    return float( (own_moves ^ 3) / opp_moves)



def heuristic_G(game, player):
    """Similar to "E": (own_moves ^4) / (opp moves^2)

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    if (opp_moves == 0 ):
        return float( (own_moves ^ 4) )
    return float( (own_moves ^ 4) / (opp_moves*opp_moves))

def heuristic_H(game, player):
    """Heuristic Based on (own_moves - opp_moves) / cells_left

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    cells_left = game.width * game.height - game.move_count

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if (cells_left == 0):
        return ((-1) * own_moves)
    else:
        return (own_moves - opp_moves) / cells_left
=======
    # TODO: finish this function!
    raise NotImplementedError


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    raise NotImplementedError
>>>>>>> origin/master


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    raise NotImplementedError


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


<<<<<<< HEAD
    def get_move(self, game, legal_moves, time_left):
=======
class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
>>>>>>> origin/master
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
<<<<<<< HEAD
        score_and_move_tuple = []
=======
>>>>>>> origin/master
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
<<<<<<< HEAD
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            move = []
            previous_best_result = []
            if (self.iterative != True):
                if (self.method == 'minimax'):
                    score_and_move_tuple.append(self.minimax(game, self.search_depth))
                elif (self.method == 'alphabeta'):
                    score_and_move_tuple.append(self.alphabeta(game, self.search_depth))
                return score_and_move_tuple[0][1]
            else: # (self.iterative = True)
                for depth in range(sys.maxsize ** 10):
                    if (self.method == 'minimax'):
                        _, move = self.minimax(game, depth+1)
                    elif (self.method == 'alphabeta'):
                        _, move = self.alphabeta(game, depth+1)
                    previous_best_result.append(move)
                if move != None:
                    return move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            if (move != (-1, -1) and move != None):
                return move
            else:
                return max(previous_best_result)

        # Return the best move from the last completed search iteration


    def max_value_MM(self, board, minimax_call_depth, depth):
        '''
        Minimax Helper function to handle the maximizing player
        function MAX-VALUE(state) returns a utility value
            if TERMINAL-TEST(state) then
                return UTILITY(state)
            v ← −∞
            for each a in ACTIONS(state) do
                    v ← MAX(v, MIN-VALUE(RESULT(s, a)))
            return v
        '''
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        minimax_call_depth = minimax_call_depth + 1
        score_and_move_tuple = []

        if (board.is_winner(self) or board.is_loser(self) or (minimax_call_depth == depth) or (len(board.get_legal_moves()) == 0)):
            return self.score(board, self)

        for move in board.get_legal_moves():
            score_and_move_tuple.append(( self.min_value_MM(board.forecast_move(move), minimax_call_depth, depth)))
        return max(score_and_move_tuple)

    def min_value_MM(self, board, minimax_call_depth, depth):
        '''
        Minimax Helper function to handle the minimizing player

        function MIN-VALUE(state) returns a utility value
            if TERMINAL-TEST(state) then
                return UTILITY(state)
            v←∞
            for each a in ACTIONS(state) do
                v ← MIN(v, MAX-VALUE(RESULT(s, a)))
            return v
        '''

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        minimax_call_depth = minimax_call_depth + 1
        score_and_move_tuple = []
        if (board.is_winner(self) or board.is_loser(self) or (minimax_call_depth == depth) or (len(board.get_legal_moves()) == 0)):
            return self.score(board, self)

        for move in board.get_legal_moves():
            score_and_move_tuple.append(( self.max_value_MM(board.forecast_move(move), minimax_call_depth, depth)))
        return min(score_and_move_tuple)

=======
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
>>>>>>> origin/master

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        minimax_call_depth = 0
        score_and_move_tuple = []
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if(len(game.get_legal_moves()) == 0):
            return(self.score(game, self), (-1, -1))

        for move in game.get_legal_moves():
            next_state = game.forecast_move(move)
            score_and_move_tuple.append((self.min_value_MM(next_state, minimax_call_depth, depth), move))

        return max(score_and_move_tuple, key=lambda t: t[0])


    def max_value_AB(self, board, alpha, beta, minimax_call_depth, depth):
        '''
        Alphabeta Helper function to handle the minimizing player
        function MAX-VALUE(state, α, β) returns a utility value
        if TERMINAL-TEST(state) then return UTILITY(state)
        v ← −∞
        for each a in ACTIONS(state) do
            v ← MAX(v, MIN-VALUE(RESULT(s,a), α, β))
            if v ≥ β then return v
            α ← MAX(α, v)
        return v
        '''
        minimax_call_depth = minimax_call_depth + 1
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        if (board.is_winner(self) or board.is_loser(self) or (minimax_call_depth == depth) or (len(board.get_legal_moves()) == 0)):
            return self.score(board, self)

        value_return = float('-inf')
        for move in board.get_legal_moves():
            value_return = max(value_return, self.min_value_AB(board.forecast_move(move), alpha, beta, minimax_call_depth, depth))
            if (value_return >= beta):
                return value_return
            alpha = max(alpha, value_return)
        return value_return

    def min_value_AB(self, board, alpha, beta, minimax_call_depth, depth):
        '''
        Alphabeta Helper function to handle the minimizing player
        function MIN-VALUE(state, α, β) returns a utility value
        if TERMINAL-TEST(state) then return UTILITY(state)
        v ← +∞
        for each a in ACTIONS(state) do
            v ← MIN(v, MAX-VALUE(RESULT(s,a) , α, β))
            if v ≤ α then return v
            β ← MIN(β, v)
        return v
        '''
        minimax_call_depth = minimax_call_depth + 1
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        if (board.is_winner(self) or board.is_loser(self) or (minimax_call_depth == depth) or (len(board.get_legal_moves()) == 0)):
            return self.score(board, self)

        value_return = float('inf')
        for move in board.get_legal_moves():
            value_return = min(value_return, self.max_value_AB(board.forecast_move(move), alpha, beta, minimax_call_depth, depth))
            if (value_return <= alpha):
                return value_return
            beta = min(beta, value_return)
        return value_return


<<<<<<< HEAD
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):

        '''
        function ALPHA-BETA-SEARCH(state) returns an action
            v ← MAX-VALUE(state,−∞,+∞)
            return the action in ACTIONS(state) with value v

        '''

        """Implement minimax search with alpha-beta pruning as described in the
        lectures.
=======

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!
        raise NotImplementedError

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
>>>>>>> origin/master

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        minimax_call_depth = 0
        score_and_move_tuple = []
        value_return = float('-inf')
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if(len(game.get_legal_moves()) == 0):
            return(self.score(game, self), (-1, -1))

        for move in game.get_legal_moves():
            next_state = game.forecast_move(move)
            value_return = self.min_value_AB(next_state, alpha, beta, minimax_call_depth, depth)
            score_and_move_tuple.append((value_return, move))
            alpha = max(alpha, value_return)

        return max(score_and_move_tuple, key = lambda t: t[0])


