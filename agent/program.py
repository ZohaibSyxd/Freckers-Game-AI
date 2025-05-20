# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction

import copy

ILLEGAL_RED_DIRECTIONS = set([
    Direction.Up,
    Direction.UpRight,
    Direction.UpLeft,
])

ILLEGAL_BLUE_DIRECTIONS = set([
    Direction.Down,
    Direction.DownRight,
    Direction.DownLeft,
])

class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Freckers game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color    
        self._board = SimpleBoard(size=8)
        self._board.initialize_board(color)

        # Initialize frogs based on the starting positions
        self._my_frogs = set(self._board.frogs[color])
        self._opponent_frogs = set(self._board.frogs[PlayerColor.RED if color == PlayerColor.BLUE else PlayerColor.BLUE])

        self._turn = color
        self._turn = self._color
        
        #print(f"Initialized agent with color: {self._color}")  # Debug print
        #print(f"My frogs: {self._my_frogs}, Opponent frogs: {self._opponent_frogs}")  # Debug print
        
    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """
        
        gamestate = GameState(
            self._board,
            self._my_frogs,
            self._opponent_frogs,
            self._color
        )
        _, best_action = minimax(gamestate, depth=3, maximizing_player=True)
        return best_action or GrowAction() # Default to GrowAction if no action is found

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Update the agent's internal state based on the action taken.
        """
        #If the action is a move, update the board and the set of frogs
        if isinstance(action, MoveAction):
            start = action.coord
            current = start
            for d in action.directions:
                current = current + d

            if color == self._color:
                self._my_frogs.discard(start)
                self._my_frogs.add(current)
            else:
                self._opponent_frogs.discard(start)
                self._opponent_frogs.add(current)

            self._board.apply_action(action, start, color)

        #If the action is a grow, update the board and the set of frogs
        elif isinstance(action, GrowAction):
            self._board.apply_action(action, None, color)

        self._turn = PlayerColor.RED if self._turn == PlayerColor.BLUE else PlayerColor.BLUE



class SimpleBoard:
    def __init__(self, size: int = 8):
        self.size = size
        self.board = [[None for _ in range(size)] for _ in range(size)]
        self.lily_pads = set() # Track lily pads
        self.frogs = {PlayerColor.RED: set(), PlayerColor.BLUE: set()}  # Track frogs for each player
        
    def initialize_board(self, player_color: PlayerColor):
        """
        Initialize the board with starting positions of frogs and lily pads.
        """
        # Set up frogs based on the player's color
        if player_color == PlayerColor.RED or player_color == PlayerColor.BLUE:
            red_frogs = {Coord(0, c) for c in range(1, 7)}
            blue_frogs = {Coord(7, c) for c in range(1, 7)}

        # Add frogs to the board
        for frog in red_frogs:
            self.board[frog.r][frog.c] = PlayerColor.RED
            self.frogs[PlayerColor.RED].add(frog)
            
        for frog in blue_frogs:
            self.board[frog.r][frog.c] = PlayerColor.BLUE
            self.frogs[PlayerColor.BLUE].add(frog)
            
        # Add lily pads to the board
        starting_positions = {Coord(0, 0), Coord(0, 7), Coord(1, 1), Coord(1, 2),
                              Coord(1, 3), Coord(1, 4), Coord(1, 5), Coord(1, 6),
                              Coord(7, 0), Coord(7, 7), Coord(6, 1), Coord(6, 2),
                              Coord(6, 3), Coord(6, 4), Coord(6, 5), Coord(6, 6)}
        
        for pos in starting_positions:
            if self.board[pos.r][pos.c] is None:
                self.board[pos.r][pos.c] = "LilyPad"
                self.lily_pads.add(pos)
        
    
    def display(self):
        """
        Display the board in a human-readable format.
        """
        for r in range(self.size):
            row = []
            for c in range(self.size):
                coord = Coord(r, c)
                if coord in self.frogs[PlayerColor.RED]:
                    row.append("R")
                elif coord in self.frogs[PlayerColor.BLUE]:
                    row.append("B")
                elif coord in self.lily_pads:
                    row.append("L")
                else:
                    row.append(".")
            print(" ".join(row))


    def apply_action(self, action: Action, frog: Coord, player_color: PlayerColor):
        """
        Apply the action to the board.
        """
        if isinstance(action, MoveAction):
            self._apply_move(frog, action, player_color)
        elif isinstance(action, GrowAction):
            self._apply_grow(player_color)


    def _apply_move(self, frog: Coord, action: MoveAction, player_color: PlayerColor):
        """
        Apply a move action for the given frog.
        Updates the board and frog positions.
        """
        opponent_color = PlayerColor.RED if player_color == PlayerColor.BLUE else PlayerColor.BLUE
        # Check if the frog is in the player's set of frogs or on a lily pad
        if (frog not in self.frogs[player_color] 
            and frog in self.frogs[opponent_color] 
            and frog in self.lily_pads):
            return

        direction = action.directions[0]
        target = frog + direction

        # Make sure move is within bounds
        if not (0 <= target.r < self.size and 0 <= target.c < self.size):
            return

        # Make sure move is to a lily pad and not occupied
        if (target not in self.lily_pads or 
            target in self.frogs[PlayerColor.RED] or 
            target in self.frogs[PlayerColor.BLUE] or
            self.board[target.r][target.c] != "LilyPad"):
            return

        # Clear current position
        self.board[frog.r][frog.c] = None
        self.lily_pads.discard(frog)  # Optional: remove lily pad under starting position

        # Move frog
        self.board[target.r][target.c] = player_color
        self.frogs[player_color].remove(frog)
        self.frogs[player_color].add(target)
        self.lily_pads.discard(target)

    def _apply_grow(self, player_color: PlayerColor):
        """
        Apply a grow action for the given player color.
        """
        for frog in self.frogs[player_color]:
            for direction in Direction:
                # Calculate the potential neighbor coordinate
                neighbor_r = frog.r + direction.value.r
                neighbor_c = frog.c + direction.value.c

                # Check if the neighbor is within bounds
                if not (0 <= neighbor_r < self.size and 0 <= neighbor_c < self.size):
                    continue

                neighbor = frog + direction
                if (self.board[neighbor.r][neighbor.c] is None 
                    and neighbor not in self.frogs[PlayerColor.RED] 
                    and neighbor not in self.frogs[PlayerColor.BLUE]
                    and neighbor not in self.lily_pads):
                    
                    self.board[neighbor.r][neighbor.c] = "LilyPad"
                    self.lily_pads.add(neighbor)
                            
    def _within_bounds(self, coord: Coord) -> bool:
        """
        Check if the coordinate is within the bounds of the board.
        """
        return 0 <= coord.r < self.size and 0 <= coord.c < self.size   
                      
    def _get_legal_moves(self, coord: Coord, player_color: PlayerColor) -> list[MoveAction]:
        """
        Get all legal moves for a given frog at `coord` for the specified `player_color`.
        """
        moves = []
        
        if coord not in self.frogs[player_color]:
            return moves
        
        # Determine illegal directions based on the player's color
        illegal_directions = ILLEGAL_RED_DIRECTIONS if player_color == PlayerColor.RED else ILLEGAL_BLUE_DIRECTIONS

        # Check all possible directions
        for direction in Direction:
            # Skip illegal directions
            if direction in illegal_directions:
                continue

            # Calculate the potential neighbor coordinate
            neighbor_r = coord.r + direction.value.r
            neighbor_c = coord.c + direction.value.c

            # Check if the neighbor is within bounds
            if not (0 <= neighbor_r < self.size and 0 <= neighbor_c < self.size):
                continue

            # Create the neighbor coordinate
            next_coord = Coord(neighbor_r, neighbor_c)

            # Check if the next coordinate is a lily pad and not occupied by a frog
            if (next_coord in self.lily_pads 
                and next_coord not in self.frogs[PlayerColor.RED] 
                and next_coord not in self.frogs[PlayerColor.BLUE]):
                
                # Valid move to an empty lily pad
                moves.append(MoveAction(coord, (direction)))

        return moves
                            
    
    def _is_grow_legal(self, player_color: PlayerColor) -> bool:
        """
        Check if grow is legal
        """
        for frog in self.frogs[player_color]:
            for direction in Direction:
                # Calculate the potential neighbor coordinate
                new_r = frog.r + direction.value.r
                new_c = frog.c + direction.value.c
            
                if 0 <= new_r < self.size and 0 <= new_c < self.size:
                    neighbor = frog + direction
                    # Check if the neighbor is within bounds and not occupied by a frog
                    if (neighbor not in self.lily_pads and
                        neighbor not in self.frogs[PlayerColor.RED] and
                        neighbor not in self.frogs[PlayerColor.BLUE]):
                        return True  # Found a valid grow position
        
        return False

                    

class GameState:
    def __init__(self, board: SimpleBoard, my_frogs: set[Coord], opponent_frogs: set[Coord], turn: PlayerColor, visited_states=None):
        self.board = copy.deepcopy(board)
        self.my_frogs = set(my_frogs)
        self.opponent_frogs = set(opponent_frogs)
        self.turn = turn
        self.visited_states = visited_states or set()
        self.visited_positions = set()

    def copy(self) -> "GameState":
        """
        Create a deep copy of the game state."""
        new_state = GameState(
            copy.deepcopy(self.board),
            set(self.my_frogs),
            set(self.opponent_frogs),
            self.turn,
            set(self.visited_states)
        )
        new_state.visited_positions = set(self.visited_positions)
        return new_state

    def get_current_player(self) -> PlayerColor:
        """
        Get the current player color.
        """
        return self.turn

    def _opponent(self, color: PlayerColor) -> PlayerColor:
        """
        Get the opponent's color.
        """
        return PlayerColor.RED if color == PlayerColor.BLUE else PlayerColor.BLUE

    def is_terminal(self) -> bool:
        """
        Check if the game state is terminal (no legal actions available).
        """
        return len(self.get_legal_actions()) == 0

    def get_legal_actions(self) -> list[Action]:
        """
        Get all legal actions for the current player.
        """
        actions = []
        for frog in self.my_frogs:
            actions.extend(self.get_legal_move_actions(frog))
        if self.is_grow_legal():
            actions.append(GrowAction())
        
        #print(f"Legal actions for {self.turn}: {actions}")  # Debug print
        #self.board.display()  # Debug print
        #print(f"self.my_frogs: {self.my_frogs}, self.opponent_frogs: {self.opponent_frogs}")  # Debug print
        #print(f"self.lilypads: {self.board.lily_pads}")  # Debug print
        return actions

    def apply_action(self, action: Action) -> "GameState":
        """
        Apply the action to the game state and return a new game state.
        """
        new_state = self.copy()

        if isinstance(action, MoveAction):
            frog = action.coord
            start = frog
            end = start
            for direction in action.directions:
                end = end + direction

            new_state.board.apply_action(action, frog, self.turn)
            new_state.my_frogs.remove(start)
            new_state.my_frogs.add(end)

        elif isinstance(action, GrowAction):
            new_state.board.apply_action(action, None, self.turn)
            # Add lily pads only; frog positions stay unchanged

        # Swap players
        new_state.my_frogs, new_state.opponent_frogs = new_state.opponent_frogs, new_state.my_frogs
        new_state.turn = self._opponent(self.turn)

        return new_state

    def is_grow_legal(self) -> bool:
        """
        Check if grow is legal for the current player.
        """
        return self.board._is_grow_legal(self.turn)

    def get_legal_move_actions(self, frog: Coord) -> list[MoveAction]:
        """
        Get all legal move actions for the given frog.
        """
        return self.board._get_legal_moves(frog, self.turn)
    
    def get_current_player(self) -> PlayerColor:
        """
        Get the current player color.
        """
        return self.turn

    def _opponent(self, color: PlayerColor) -> PlayerColor:
        """
        Get the opponent's color.
        """
        return PlayerColor.RED if color == PlayerColor.BLUE else PlayerColor.BLUE

    def is_terminal(self) -> bool:
        return len(self.get_legal_actions()) == 0

    def get_legal_actions(self) -> list[Action]:
        """
        Get all legal actions for the current player.
        """
        actions = []
        for frog in self.my_frogs:
            actions.extend(self.get_legal_move_actions(frog))
        if self.is_grow_legal():
            actions.append(GrowAction())
        return actions

    def apply_action(self, action: Action) -> "GameState":
        """
        Apply the action to the game state and return a new game state.
        """
        new_state = self.copy()
        if isinstance(action, MoveAction):
            start = action.coord
            end = start
            for direction in action.directions:
                end = end + direction
            new_state.board.apply_action(action, start, self.turn)
            new_state.my_frogs.remove(start)
            new_state.my_frogs.add(end)
        elif isinstance(action, GrowAction):
            new_state.board.apply_action(action, None, self.turn)

        new_state.my_frogs, new_state.opponent_frogs = new_state.opponent_frogs, new_state.my_frogs
        new_state.turn = self._opponent(self.turn)
        return new_state

    def is_grow_legal(self) -> bool:
        return self.board._is_grow_legal(self.turn)

    def get_legal_move_actions(self, frog: Coord) -> list[MoveAction]:
        return self.board._get_legal_moves(frog, self.turn)

    def evaluate(self) -> float:
        my_mobility = sum(len(self.get_legal_move_actions(f)) for f in self.my_frogs)
        opp_mobility = sum(len(self.board._get_legal_moves(f, self._opponent(self.turn))) for f in self.opponent_frogs)
        mobility_score = (my_mobility - opp_mobility) * 2

        def total_min_distance(from_set, to_set):
            return sum(min(abs(a.r - b.r) + abs(a.c - b.c) for b in to_set) for a in from_set) if from_set and to_set else 0

        pressure_score = -0.2 * total_min_distance(self.my_frogs, self.opponent_frogs)

        def advancement(frogs):
            if not frogs:
                return 0
            total = sum(frog.r for frog in frogs)
            return total / len(frogs) if self.turn == PlayerColor.RED else -total / len(frogs)

        advancement_score = (advancement(self.my_frogs) - advancement(self.opponent_frogs)) * 2

        def centrality(frog):
            mid = self.board.size // 2
            return -((frog.r - mid) ** 2 + (frog.c - mid) ** 2)

        board_control_score = sum(centrality(f) for f in self.my_frogs) * 0.05

        def average_distance_within_group(frogs):
            frogs = list(frogs)
            n = len(frogs)
            if n <= 1:
                return 0
            total = sum(
                abs(frogs[i].r - frogs[j].r) + abs(frogs[i].c - frogs[j].c)
                for i in range(n) for j in range(i + 1, n)
            )
            return total / (n * (n - 1) / 2)

        cluster_score = -0.1 * average_distance_within_group(self.my_frogs)

        def count_threats(my_frogs, opp_frogs):
            return sum(
                1 for m in my_frogs
                for o in opp_frogs
                if abs(m.r - o.r) + abs(m.c - o.c) == 1
            )

        threat_score = -0.5 * count_threats(self.my_frogs, self.opponent_frogs)

        def edge_penalty(frog):
            if frog.r in (0, self.board.size - 1) or frog.c in (0, self.board.size - 1):
                return -1
            return 0

        defensive_score = sum(edge_penalty(f) for f in self.my_frogs) * 0.5

        total = (
            mobility_score + pressure_score + advancement_score +
            board_control_score + cluster_score + threat_score + defensive_score
        )

        # Repetition check
        position_signature = tuple(sorted((f.r, f.c) for f in self.my_frogs))
        if position_signature in self.visited_positions:
            total -= 200
        self.visited_positions.add(position_signature)

        if position_signature in self.visited_states:
            total -= 5
        else:
            self.visited_states.add(position_signature)

        def net_progress(frog):
            return frog.r if self.turn == PlayerColor.RED else -frog.r

        displacement_score = sum(net_progress(f) for f in self.my_frogs) * 2
        total += displacement_score

        return total


def score_action(state: GameState, action: Action) -> float:
    if isinstance(action, MoveAction):
        total_dir_r = sum(d.r for d in action.directions)
        score = total_dir_r if state.turn == PlayerColor.RED else -total_dir_r

        dest = action.coord
        for d in action.directions:
            dest += d
        if dest in state.opponent_frogs:
            score += 10
        return score
    elif isinstance(action, GrowAction):
        return 0
    return -1000

    
def minimax(state: GameState, depth: int, maximizing_player: bool, alpha=float('-inf'), beta=float('inf')) -> tuple[float, Action | None]:
    if depth == 0 or state.is_terminal():
        eval_score = state.evaluate()
        # Apply a depth-based penalty
        eval_score -= (2 - depth) * 0.1  # Penalize deeper repetitions
        return eval_score, None

    legal_actions = state.get_legal_actions()

    scored_actions = [(score_action(state, a), a) for a in legal_actions]
    scored_actions.sort(key=lambda x: x[0], reverse=maximizing_player)
    
    best_action = None

    if maximizing_player:
        max_eval = float('-inf')
        for _, action in scored_actions:
            next_state = state.apply_action(action)
            eval, _ = minimax(next_state, depth - 1, False, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_action = action
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # β cut-off
        return max_eval, best_action

    else:
        min_eval = float('inf')
        for _, action in scored_actions:
            next_state = state.apply_action(action)
            eval, _ = minimax(next_state, depth - 1, True, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_action = action
            beta = min(beta, eval)
            if beta <= alpha:
                break  # α cut-off
        return min_eval, best_action