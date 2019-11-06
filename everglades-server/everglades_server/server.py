import os
import json
import csv
import datetime
import pdb

import numpy as np

from everglades_server.definitions import *

class EvergladesGame:
    """ 
    """ 
    def __init__(self, **kwargs):
        # Get configuration locations
        config_path = kwargs.get('config_dir')
        map_file = kwargs.get('map_file')
        unit_file = kwargs.get('unit_file')
        self.debug = kwargs.get('debug', False)
        self.view = kwargs.get('view', False)
        self.createOut = kwargs.get('out', False)
        self.player_names = kwargs.get('pnames')
        self.output_dir = kwargs.get('output_dir')
        #config_file = kwargs.get(

        # Initialize game
        if os.path.exists(map_file):
            self.board_init(map_file)
        elif os.path.exists(config_dir + map_file):
            self.board_init(map_file)
        else:
            # Exit with error
            pass

        if os.path.exists(unit_file):
            self.unitTypes_init(unit_file)
        elif os.path.exists(config_dir + unit_file):
            self.unitTypes_init(unit_file)
        else:
            # Exit with error
            pass

        # Check output directory existance. Create if necessary
        if not os.path.isdir(self.output_dir):
            oldmask = os.umask(000)
            os.mkdir(self.output_dir,mode=0o777)
            os.umask(oldmask)
        assert( os.path.isdir(self.output_dir) ), 'Output directory does not exist \
                and could not be created'


        # Initialize output arrays, to be written to file at game completion
        # Needs the map name to be populated before initialization
        self.output_init()

    def board_init(self,map_file):
        """ 
        """ 
        ## Game board initialization
        # Load in map json configuration file
        with open(map_file) as fid:
            self.map_dat = json.load(fid)

        # Initialize map
        self.evgMap = EvgMap(self.map_dat['MapName'])

        self.team_starts = {}
        # Two different types of map keys. Note that they may both be the same
        #   1) Array sorted by index of evgMap node with corresponding node id
        #   2) Array sorted by node id with corresponding index of evgMap node
        self.map_key1 = np.zeros( len(self.map_dat['nodes']), dtype=np.int )
        # Create the map nodes 
        for i, in_node in enumerate(self.map_dat['nodes']):
            # Initialize node
            node = EvgMapNode(
                    ID = in_node['ID'],
                    radius = in_node['Radius'],
                    resource = in_node['Resource'],
                    defense = in_node['StructureDefense'],
                    points = in_node['ControlPoints'],
                    teamStart = in_node['TeamStart']
            )
            if node.teamStart != -1:
                self.team_starts[node.teamStart] = node.ID
            # Add node connections
            for in_conn in in_node['Connections']:
                # Initialize node outbound connections
                conn = EvgNodeConnection(
                        destID = in_conn['ConnectedID'],
                        distance = in_conn['Distance']
                )
                node.connections.append(conn)
                node.connection_idxs.append(conn.destID)
            # end connection creation
            node.connection_idxs = np.array(node.connection_idxs).astype(np.int)
            
            self.map_key1[i] = in_node['ID']
            # Append node to the map
            self.evgMap.nodes.append(node)
        # end node creation
        self.map_key2 = np.argsort( self.map_key1 )

        # Convert p0 nodes numbering to p1
        # Need method to do this when boards are not hand-designed
        self.p1_node_map = [0, 11, 8, 9, 10, 5, 6, 7, 2, 3, 4, 1] # DemoMap mapping
        
        def _convert_node(node_num):
            return self.p1_node_map[int(node_num)]
        
        self._vec_convert_node = np.vectorize(_convert_node)

        # Fortress defense multiplier for controlling player's units
        # located at the fortress node
        self.fort_bonus = 2
        # Watchtower vision bonus. Extra graph depth of fog of war penetration
        self.watch_bonus = 1


    def unitTypes_init(self,unit_file):
        """ 
        """ 
        ## Unit types initialization
        # Load in unit types json configuration file
        with open(unit_file) as fid:
            self.unit_dat = json.load(fid)

        # Initialize unit types
        self.unit_types = []
        uid = 0
        self.unit_ids = {}
        self.unit_names = {}
        for in_type in self.unit_dat['units']:
            # Initialize new unit type
            unit_type = EvgUnitDefinition(
                    name = in_type['Name'],
                    health = in_type['Health'],
                    damage = in_type['Damage'],
                    speed = in_type['Speed'],
                    control = in_type['Control'],
                    cost = in_type['Cost']
            )
            self.unit_types.append(unit_type)
            #pdb.set_trace()
            self.unit_ids[uid] = unit_type.unitType.lower()
            self.unit_names[unit_type.unitType.lower()] = uid
            uid += 1
        # end unit type creation

    def game_init(self, player_dat):
        """ 
        """ 
        # Open up connections
        # Wait for two players
        # Assign player numbers
        # Initialize players and units

        self.current_turn = 0
        self.battleField = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
        self.players = {}

        #pdb.set_trace()
        players = list(player_dat.keys())
        # Where to set a random player 0/player 1; here?
        # ...is it even necessary? They both think they start at node 1.
        #np.random.choice(players)
        # Cumulative group ID for output
        map_gid = 1
        # Cumulative unit number for output
        map_units = 1

        for player in players:
            assert(player in self.team_starts), 'Given player number not included in map configuration file starting locations'
            start_node_idx = self.team_starts[player]
            map_node_idx = np.argwhere(self.map_key1 == start_node_idx)[0][0]

            self.players[player] = EvgPlayer(player)
            for i, gid in enumerate(player_dat[player]['unit_config']):
                in_type, in_count = player_dat[player]['unit_config'][gid]
                in_type = in_type.lower()

                # Input validation
                assert(in_type in self.unit_names), 'Group type not in unit type config file'
                assert(in_count <= 100), 'Invalid group size allocation'
                # TODO: create a cost counter to make sure the total unit allocation is correct

                unit_id = self.unit_names[in_type]
                newGroup = EvgGroup(
                        groupID = gid,
                        location = start_node_idx,
                        mapGroupID = map_gid,
                        mapUnitID = map_units
                )
                outtype = in_type[0].upper() + in_type[1:]
                # BUG - will only work if there is one unit type per group; fine for now
                outstr = '{:.6f},{},{},{},[{}],[{}],[{}]'.format(self.current_turn,
                                                                 player,
                                                                 map_gid,
                                                                 start_node_idx,
                                                                 outtype,
                                                                 map_units,
                                                                 in_count
                )
                self.output['GROUP_Initialization'].append(outstr)
                map_gid += 1
                map_units += in_count

                newUnit = EvgUnit(
                        unitType = in_type,
                        count = in_count
                )
                newUnit.definition = self.unit_types[ self.unit_names[in_type] ]
                newGroup.units.append(newUnit)
                self.players[player].groups.append(newGroup)                    
                self.evgMap.nodes[map_node_idx].groups[player].append(i)
            # end group loop
        # end player loop
        #pdb.set_trace()

        self.total_groups = map_gid
        self.total_units = map_units
        self.focus = np.random.randint(self.total_groups)
        self.capture()
        self.game_end() # To output initial score for time 0

        return

    def game_turn(self, actions):
        """ 
        """ 
        self.current_turn += 1

        ## Apply each player's turn
        #pdb.set_trace()
        for player in self.team_starts:
            if player not in actions:
                print('Player {} not found in input action dictionary'.format(player))
                continue

            action = actions[player]
            # Verify valid shape of action array
            r,c = action.shape[:2]
            assert(c == 2), 'Did not receive 2 columns for player {}s action'.format(player)
            action = action[:7,:] # max 7 actions per turn

            # Verfiy each swarm gets commanded only once
            used_swarms = []
            # group id, node id
            for gid, nid in action.astype(int):
                if player == 1:
                    nid = int( self._vec_convert_node(nid) )
                current_node = self.players[player].groups[gid].location
                map_idx = np.where(self.map_key1 == current_node)[0][0]

                #pdb.set_trace()
                ## Tests
                # Ensure this swarm hasn't been commanded this turn
                test1 = gid not in used_swarms
                # Ensure swarm isn't already in transit
                test2 = self.players[player].groups[gid].moving == False
                # Ensure the new node is connected to the current node
                test3 = False
                for conn in self.evgMap.nodes[map_idx].connections:
                    if conn.destID == nid:
                        test3 = True
                        distance = conn.distance
                        break
                if test1 and test2 and test3:
                    used_swarms.append(gid)
                    #print('good move')
                    outstr = '{:.6f},{},{},{},{},{}'.format(
                            self.current_turn,
                            player,
                            self.players[player].groups[gid].groupID,
                            self.players[player].groups[gid].location,
                            nid,
                            'RDY_TO_MOVE'
                    )
                    self.output['GROUP_MoveUpdate'].append(outstr) 

                    self.players[player].groups[gid].ready = True
                    self.players[player].groups[gid].moving = False
                    self.players[player].groups[gid].travel_destination = nid
                    self.players[player].groups[gid].distance_remaining = distance
            # end action application loop
        # end player loop

        self.combat()
        self.movement()
        self.capture()
        self.build_knowledge_output()

        return self.game_end() # returns scores and status

    def game_end(self):

        # Game end types
        end_states = {}
        end_states['InProgress']   = 0 
        end_states['TimeExpired']  = 1 
        end_states['BaseCapture']  = 2 
        end_states['Annihilation'] = 3 
        status = end_states['InProgress']

        scores = {i:0 for i in self.team_starts}
        counts = [0 for i in self.team_starts]
        base_captured = [0 for i in self.team_starts]
        home_loss = {i:0 for i in self.team_starts}
        pids = np.array( list(self.players.keys()) )

        # Add node points to player scores
        for i, node in enumerate(self.evgMap.nodes):
            if (node.teamStart != -1) and \
               (node.controlledBy != -1) and \
               (node.controlledBy != node.teamStart):
                # Extra bonus for capturing the opponent's base
                base_captured[node.teamStart] = 1 
                scores[node.controlledBy] += 1000
            if node.controlState != 0:
                # Points for controlling or partially controlling a node
                pid = 0 if node.controlState > 0 else 1
                xer = 2 if np.abs(node.controlState) == node.controlPoints else 1
                points = node.controlPoints if xer == 2 else np.abs(node.controlState)
                scores[pid] += np.abs( points * xer )

        # Add unit points to player scores
        for i, pid in enumerate(self.team_starts):
            for i, group in enumerate(self.players[pid].groups):
                if group.destroyed == False:
                    counts[pid] += np.sum( [i.count for i in group.units] )
                    scores[pid] += np.sum( [(i.count * i.definition.cost) for i in group.units] )

        ## Check progress
        # Time expiration
        if self.current_turn >= 150:
            status = end_states['TimeExpired']
        # Annihilation
        elif np.sum(counts) == 0:
            status = end_states['Annihilation']
        # Base capture
        elif 1 in base_captured:
            status = end_states['BaseCapture']

        if status != 0:
            outstr = '{:.6f},{},{}'.format(
                    self.current_turn,
                    self.player_names[0],
                    self.player_names[1]
            )
            self.output['PLAYER_Tags'].append(outstr) 

        if (self.current_turn % 10 == 0):
            self.focus = np.random.randint(self.total_groups)

        outstr = '{:.6f},{},{},{},{}'.format(
                self.current_turn,
                scores[0],
                scores[1],
                status,
                self.focus
        )
        self.output['GAME_Scores'].append(outstr) 

        if status != 0 and self.createOut == 1:
            self.write_output()

        return scores, status


    def view_state(self):
        print(f'\t- View turn {self.current_turn} -\n')
        def battleField_Update(self, node, control):
            if node == 1:
                self.battleField[1][0] = control
            
            if node == 2:
                self.battleField[0][1] = control

            if node == 3:
                self.battleField[1][1] = control
            
            if node == 4:
                self.battleField[2][1] = control

            if node == 5:
                self.battleField[0][2] = control
            
            if node == 6:
                self.battleField[1][2] = control

            if node == 7:
                self.battleField[2][2] = control
            
            if node == 8:
                self.battleField[0][3] = control

            if node == 9:
                self.battleField[1][3] = control

            if node == 10:
                self.battleField[2][3] = control
            
            if node == 11:
                self.battleField[1][4] = control

        # For each node
        for i, nidx in enumerate(self.map_key2):
            battleField_Update(self, self.evgMap.nodes[nidx].ID, self.evgMap.nodes[nidx].controlState)

        print(f'       ({self.battleField[0][1]:>4d}) ({self.battleField[0][2]:>4d}) ({self.battleField[0][3]:>4d})')
        print(f'      /                    \\')
        print(f'({self.battleField[1][0]:>4d}) ({self.battleField[1][1]:>4d}) ({self.battleField[1][2]:>4d}) ({self.battleField[1][3]:>4d}) ({self.battleField[1][4]:>4d})')
        print(f'      \\                    /')
        print(f'       ({self.battleField[2][1]:>4d}) ({self.battleField[2][2]:>4d}) ({self.battleField[2][3]:>4d})')
        print(f'\n\n')

    def debug_state(self):
        print(f'\t - Debug turn {self.current_turn} -\n')

        def battleField_Update(self, node, control):
            if node == 1:
                self.battleField[1][0] = control
            
            if node == 2:
                self.battleField[0][1] = control

            if node == 3:
                self.battleField[1][1] = control
            
            if node == 4:
                self.battleField[2][1] = control

            if node == 5:
                self.battleField[0][2] = control
            
            if node == 6:
                self.battleField[1][2] = control

            if node == 7:
                self.battleField[2][2] = control
            
            if node == 8:
                self.battleField[0][3] = control

            if node == 9:
                self.battleField[1][3] = control

            if node == 10:
                self.battleField[2][3] = control
            
            if node == 11:
                self.battleField[1][4] = control

        # For each node
        for i, nidx in enumerate(self.map_key2):
            print(f'Node {self.evgMap.nodes[nidx].ID}')

            print(f'\t{self.evgMap.nodes[nidx].resource}')

            print(f'\t% Controlled: {self.evgMap.nodes[nidx].controlState}')

            # Show player information
            counts = []
            cnt = 0
            for gid in self.evgMap.nodes[nidx].groups[0]:
                if self.players[0].groups[gid].moving == False:
                    cnt += self.players[0].groups[gid].units[0].count
                    counts.append(gid)

            print(f'\tPlayer 0 units: {cnt}')
            for gid in counts:
                print(f'\t\ttype: {self.players[0].groups[gid].units[0].unitType}')
                #print(f'\t\tavg health: {np.average(self.players[0].groups[gid].units[0].unitHealth)}')
                print(f'\t\t{np.floor(self.players[0].groups[gid].units[0].unitHealth)}')

            counts = []
            cnt = 0
            for gid in self.evgMap.nodes[nidx].groups[1]:
                if self.players[1].groups[gid].moving == False:
                    cnt += self.players[1].groups[gid].units[0].count
                    counts.append(gid)

            print(f'\tPlayer 1 units: {cnt}')
            for gid in counts:
                print(f'\t\ttype: {self.players[1].groups[gid].units[0].unitType}')
                #print(f'\t\tavg health: {np.average(self.players[1].groups[gid].units[0].unitHealth)}')
                print(f'\t\t{np.floor(self.players[1].groups[gid].units[0].unitHealth)}')

            print(f'\n')

        print(f'--------------------------------------------------------\n\n')


    def board_state(self, player_num):
        """ 
         |  Return the state of the game board. Returned shall be a numpy array
         |  with the following index values:
         |    Index 0
         |        0 : turn number
         |    Indices 1-4: node states. Repeats for number of nodes, starting at 5
         |        1 : boolean 'has fortress bonus'
         |        2 : boolean 'has watchtower bonus'
         |        3 : percent controlled [-100:100] player 0 owned = +, player 1 = -
         |        4 : number of opposing player units
        """ 
        assert( isinstance(player_num, (int, float)) ), '"player_num" was not a number'
        assert(player_num in self.team_starts), 'Given player number not included in map configuration file starting locations'

        # Only works for two players
        players = np.array( list(self.players.keys()) )
        opp_pid = np.where( players != player_num )[0][0]

        # Build valid indices for fog of war
        if self.debug:
            valid_nodes = [True for i in self.map_key2]
        else:
            # Assume False, prove otherwise
            valid_nodes = [False for i in self.map_key2]

            # Loop through nodes to check for validity
            for i, nidx in enumerate(self.map_key2):
                node = self.evgMap.nodes[nidx]
                if node.controlledBy == player_num:
                    # Always can see nodes you control
                    valid_nodes[i] = True
                    # Can also see connections if it has a watchtower bonus
                    if 'OBSERVE' in node.resource:
                        for j, cid in enumerate(node.connection_idxs):
                            cidx = int(np.squeeze(np.where(self.map_key1 == cid)))
                            valid_nodes[cidx] = True
                else:
                    # Must have active groups in the area to see other nodes
                    for gid in node.groups[player_num]:
                        if self.players[player_num].groups[gid].moving == False:
                            valid_nodes[i] = True
                            break
        # end fog of war masking
        #pdb.set_trace()


        num_nodes = len(self.map_key2)
        state = np.zeros(num_nodes * 4 + 1, dtype = np.int)
        idx = 0
        state[idx] = self.current_turn
        idx += 1
        # Loop through nodes from smallest to largest (hence map_key2)
        for i, nidx in enumerate(self.map_key2):
            # Flip board so both players think they start at 1
            if player_num == 1:
                node_id = self._vec_convert_node( self.map_key1[nidx] )
                nidx = int(np.squeeze(np.where(self.map_key1 == node_id)))

            node = self.evgMap.nodes[nidx]
            state[idx] = 1 if 'DEFENSE' in node.resource else 0
            state[idx+1] = 1 if 'OBSERVE' in node.resource else 0
            state[idx+2] = node.controlState

            for gid in node.groups[opp_pid]:
                group = self.players[opp_pid].groups[gid]
                for unit in group.units:
                    state[idx+3] += unit.count

            idx += 4
        # end per-node state build
        #pdb.set_trace()

        return state

    def player_state(self, player_num):
        """ 
         |  Return the state of the player's groups. Return shall be a numpy array
         |  with the following index values:
         |    Index 0
         |        0 : turn number
         |    Indices 1-5: node states. Repeats for number of nodes, starting at 6
         |        1 : location as node number (int)
         |        2 : unit type
         |        3 : average group health [0-100]
         |        4 : boolean 'is group moving'
         |        5 : number of units alive (int)
        """ 
        player = self.players[player_num]
        state = np.zeros( len(player.groups) * 5 + 1, dtype = np.int)
        idx = 0
        state[idx] = self.current_turn
        idx += 1
        for group in player.groups:
            # Build unit statistics
            units_alive = 0
            health = 0
            for unit in group.units:
                units_alive += np.sum( unit.unitHealth > 0 )
                health += np.sum( unit.unitHealth )

            location = group.location
            # Flip board so both players think they start at 1
            if player_num == 1:
                location = int( self._vec_convert_node(location) )

            state[idx] = location
            # Following line assumes one unit type per group; not necessarily true
            state[idx+1] = self.unit_names[ group.units[0].unitType.lower() ]
            state[idx+2] = (health * 1.) / units_alive if (units_alive > 0) else 0
            state[idx+3] = 1 if group.moving else 0
            state[idx+4] = units_alive

            idx += 5
        # end per-group state build

        if player_num == 2:
            pdb.set_trace()

        return state

    def combat(self):
        ## Apply combat
        # Combat occurs before movement - a fleeing group could still be within
        # targeting range during the same turn; arriving units need to get
        # bearings before attacking anything.
        all_dmg = {}

        for node in self.evgMap.nodes:
            player_gids = {}
            counts = {}
            tgt_gids = {}

            # Determine which players occupy this node
            for player in self.team_starts:
                #pdb.set_trace()
                if len(node.groups[player]) > 0:
                    player_gids[player] = []
                    counts[player] = []

                    # Build a list of groups that are available for combat
                    for gid in node.groups[player]:
                        # Discount groups in transit
                        if self.players[player].groups[gid].moving == False:
                            player_gids[player].append(gid)
                            # BUG - if group consists of different units combat is not applied to all
                            unit = self.players[player].groups[gid].units[0]
                            count = np.sum( unit.unitHealth > 0 )
                            counts[player].append(count)

                    # Remove empty list to make combat application work
                    if len(player_gids[player]) == 0:
                        player_gids.pop(player)
                    # end group loop
            # end player loop

            # Only enter combat if previous conditions hold true
            if len(player_gids) >= 2:
                #pdb.set_trace()
                # Build a damage dictionary 
                #   keys = player ids
                #   values = array with the opposing unit id that each unit targeted
                infliction = {} # damage to other units
                nulled_ids = {}
                pids = np.array( list(self.players.keys()) )

                # Build damage 
                for pid in player_gids:
                    # Only works for two players right now
                    opp_pid = np.where( pids != pid )[0][0]
                    opp_player_units = np.sum( counts[opp_pid] )
                    player_units = np.sum( counts[pid] )

                    infliction[pid] = {}
                    nulled_ids[pid] = {}
                    #pdb.set_trace()
                    for i, gid in enumerate(player_gids[pid]):
                        unittype = self.players[pid].groups[gid].units[0]
                        nulled_ids[pid][i] = []
                        for j in range(counts[pid][i]):
                                uid = np.random.randint(opp_player_units)
                                if uid in infliction[pid]:
                                    infliction[pid][uid] += unittype.definition.damage
                                else:
                                    infliction[pid][uid] = unittype.definition.damage
                # end player loop
                #pdb.set_trace()
            
                all_dmg[node.ID] = {pid:{'groups':[], 'units':[], 'health':[]} for pid in self.team_starts}
                
                # Apply damage - separate so units don't die before they get to apply their damage
                for pid in player_gids:
                    opp_pid = np.where( pids != pid )[0][0]

                    #pdb.set_trace()
                    # Loop through the infliction array
                    for tgt_idx in sorted( infliction[pid].keys() ):
                        # Determine which opposing group was affected
                        tgt_dmg = infliction[pid][tgt_idx]
                        tgt_group = 0
                        found = False

                        while found == False:
                            if tgt_idx < counts[opp_pid][tgt_group]:
                                #pdb.set_trace()
                                found = True
                                # Determine the opposing unit that was affected
                                tgt_gid = player_gids[opp_pid][tgt_group]
                                group = self.players[opp_pid].groups[tgt_gid]
                                tgt_unit = self.players[opp_pid].groups[tgt_gid].units[0]
                                tgt_armor = tgt_unit.definition.health
                                tgt_cntrl = 1 if node.controlledBy == opp_pid else 0
            
                                fort_bns = 1 if ('DEFEND' in node.resource) else 0
                                strct_def = node.defense
                                node_def = (tgt_cntrl + fort_bns) * strct_def
                                
                                # Determine damage. Damage equation: 
                                # ( 10 * damage infliction ) / ( (unit armor + fortress bonus) * structure defense )
                                loss = (10. * tgt_dmg) / (tgt_armor + node_def)

                                # Subtract damage 
                                #pdb.set_trace()
                                if (len(nulled_ids[opp_pid][tgt_group]) > 0):
                                    #pdb.set_trace()
                                    tgt_idx -= len(nulled_ids[opp_pid][tgt_group])
                                tgt_unit_idx = np.argwhere( tgt_unit.unitHealth > 0 )[tgt_idx]
                                tgt_unit.unitHealth[tgt_unit_idx] -= loss

                                outgroup = group.mapGroupID
                                outunit = group.mapUnitID + 1 + tgt_unit_idx

                                # Remove from node groups if dead
                                if tgt_unit.unitHealth[tgt_unit_idx] <= 0:
                                    outhealth = 0.
                                    tgt_unit.unitHealth[tgt_unit_idx] = 0
                                    tgt_unit.count -= 1
                                    # Does not append the original tgt_idx, but we're
                                    # fine because the tgt_idxs were sorted.
                                    nulled_ids[opp_pid][tgt_group].append(tgt_idx)
                                    # Disband the group
                                    if tgt_unit.count == 0:
                                        #pdb.set_trace()
                                        self.players[opp_pid].groups[tgt_gid].destroyed = True
                                        pop_idx = node.groups[opp_pid].index(tgt_gid)
                                        node.groups[opp_pid].pop(pop_idx)

                                        outstr = '{:.6f},{},{}'.format(
                                                self.current_turn,
                                                opp_pid,
                                                self.players[opp_pid].groups[tgt_gid].mapGroupID
                                        )
                                        self.output['GROUP_Disband'].append(outstr) 
                                else:
                                    outhealth = tgt_armor * (tgt_unit.unitHealth[tgt_unit_idx] / 100.)
                                outhealthstr = '{:.6f}'.format(float(outhealth))
                                all_dmg[node.ID][opp_pid]['groups'].append(int(outgroup))
                                all_dmg[node.ID][opp_pid]['units'].append(int(outunit))
                                all_dmg[node.ID][opp_pid]['health'].append(float(outhealthstr))
                            else:
                                tgt_idx -= counts[opp_pid][tgt_group]
                                tgt_group += 1
                        # end target unit search
                    # end target loop

                    # Build combat output message
                    outstr = '{:.6f},{},{},[{}],[{}],[{}]'.format(
                            self.current_turn,
                            opp_pid,
                            node.ID,
                            ';'.join(str(i) for i in all_dmg[node.ID][opp_pid]['groups']),
                            ';'.join(str(i) for i in all_dmg[node.ID][opp_pid]['units']),
                            ';'.join(str(i) for i in all_dmg[node.ID][opp_pid]['health']),
                    )
                    self.output['GROUP_CombatUpdate'].append(outstr) 
                            
                # end player loop
                #pdb.set_trace()
            # end if combat check
        # end node loop

    def movement(self):
        ## Apply group movements
        for player in self.team_starts:
            for group in self.players[player].groups:
                if not group.destroyed:
                    if group.ready:
                        # Let a turn pass to make Unreal logic happy
                        group.ready = False
                        group.moving = True
                    elif group.moving:
                        # Apply amount moved
                        # BUG - if group consists of different unit types, it won't move properly
                        group.distance_remaining -= group.units[0].definition.speed

                        # Get information for adjustments
                        start_idx = int( np.squeeze(np.where(self.map_key1 == group.location)) )
                        end_idx = int( np.squeeze(np.where(self.map_key1 == group.travel_destination)) )

                        # Check for arrival
                        if group.distance_remaining <= 0:
                            # ARRIVED
                            # Adjust locations and groups at each node
                            outstr = '{:.6f},{},{},{},{},{}'.format(
                                    self.current_turn,
                                    player,
                                    group.groupID,
                                    self.evgMap.nodes[start_idx].ID,
                                    self.evgMap.nodes[end_idx].ID,
                                    'ARRIVED'
                            )
                            self.output['GROUP_MoveUpdate'].append(outstr) 

                            self.evgMap.nodes[start_idx].groups[player].remove(group.groupID)
                            self.evgMap.nodes[end_idx].groups[player].append(group.groupID)
                            group.distance_remaining = 0
                            group.moving = False
                            group.location = group.travel_destination
                            group.travel_destination = -1

                        else:
                            # IN_TRANSIT
                            outstr = '{:.6f},{},{},{},{},{}'.format(
                                    self.current_turn,
                                    player,
                                    group.groupID,
                                    self.evgMap.nodes[start_idx].ID,
                                    self.evgMap.nodes[end_idx].ID,
                                    'IN_TRANSIT'
                            )
                            self.output['GROUP_MoveUpdate'].append(outstr) 

                    # end move adjustments
            # end group loop
        # end player loop

    def capture(self):
        for node in self.evgMap.nodes:
            controllers = []
            points = {}
            # Check for number of current groups at each node
            for pid in node.groups:
                points[pid] = 0
                if len(node.groups[pid]) > 0:

                    ctr = 0
                    for gid in node.groups[pid]:
                        # Discount in-transit groups
                        if self.players[pid].groups[gid].moving == False:
                            ctr += 1
                            count = self.players[pid].groups[gid].units[0].count
                            xer = self.players[pid].groups[gid].units[0].definition.control
                            points[pid] += count * xer 
                    if ctr >= 1:
                        controllers.append(pid)

            # If only 1 group, let them capture
            if len(controllers) == 1:

                if (np.abs(node.controlState) < node.controlPoints) or \
                   (controllers[0] != node.controlledBy):
                    # Logistics
                    if controllers[0] == 0:
                        pxer = 1 
                        pid = 0 
                    else:
                        pxer = -1
                        pid = 1

                    # Capture
                    #pdb.set_trace()
                    neutralize = False
                    if self.current_turn == 0:
                        node.controlState = node.controlPoints * pxer
                    else:
                        oldSign = int(node.controlState < 0)
                        node.controlState += points[pid] * pxer
                        newSign = int(node.controlState < 0)
                        neutralize = True if oldSign != newSign else False

                        # Build output
                        fullctrl = 'true' if np.abs(node.controlState) >= node.controlPoints else 'false'
                        outstr = '{:.6f},{},{},{:.6f},{}'.format(
                                self.current_turn,
                                node.ID,
                                pid,
                                np.abs(node.controlState),
                                fullctrl
                        )
                        self.output['NODE_ControlUpdate'].append(outstr) 

                    # Update
                    if np.abs(node.controlState) >= node.controlPoints:
                        node.controlState = node.controlPoints * pxer
                        node.controlledBy = pid
                    if node.controlledBy != -1 and neutralize:
                        print('!!!!!!Neutralize!!!!!!!!')
                        print(node.controlledBy)
                        node.controlledBy = -1
                        print(node.controlledBy)
                        print()


    def output_init(self):
        # Output telemetry files
        date = datetime.datetime.today()
        date_frmt = date.strftime('%Y.%m.%d-%H.%M.%S')
        self.dat_dir = self.output_dir + '/' + self.evgMap.name + '_' + date_frmt

        oldmask = os.umask(000)
        os.mkdir(self.dat_dir,mode=0o777)
        os.umask(oldmask)
        assert( os.path.isdir(self.dat_dir) ), 'Could not create telemetry output directory'

        self.output = {}
        hdr = '0,player1,player2,status,focus'
        self.output['GAME_Scores'] = [hdr]

        hdr = '0,player,node,groups,units,health'
        self.output['GROUP_CombatUpdate'] = [hdr]

        hdr = '0,player,group'
        self.output['GROUP_Disband'] = [hdr]

        hdr = '0,player,group,node,types,start,count'
        self.output['GROUP_Initialization'] = [hdr]

        hdr = '0,player,unitTypes,unitCount,status,node1,node2'
        self.output['GROUP_Knowledge'] = [hdr]

        hdr = '0,player,group,start,destination,status'
        self.output['GROUP_MoveUpdate'] = [hdr]

        hdr = '0,player,node,faction,controlvalue,controlled'
        self.output['NODE_ControlUpdate'] = [hdr]

        hdr = '0,player,nodes,knowledge,controller,percent'
        self.output['NODE_Knowledge'] = [hdr]

        hdr = '0,player1,player2'
        self.output['PLAYER_Tags'] = [hdr]

    def build_knowledge_output(self):
        players = np.array( list(self.players.keys()) )

        for pid in self.team_starts:
            opp_pid = np.where( players != pid )[0][0]
            knowledge = [0 for i in self.map_key2]
            nodes = []
            controller = []
            percent = []

            # Node knowledge
            for i, nidx in enumerate(self.map_key2):
                node = self.evgMap.nodes[nidx]
                stationed = False
                partial_nodes = []
                for gid in node.groups[pid]:
                    group = self.players[pid].groups[gid]
                    if group.moving == False:
                        stationed = True
                    else:
                        dest = group.travel_destination
                        if dest not in partial_nodes:
                            partial_nodes.append(dest)
                # end node group loop
                adj_watchtower = False
                incoming_units = False

                for j, cid in enumerate(node.connection_idxs):
                    cidx = int(np.squeeze(np.where(self.map_key1 == cid)))
                    conn = self.evgMap.nodes[cidx]

                    # See if adjacent watchtower
                    if ('OBSERVE' in conn.resource) and \
                       (conn.controlledBy == pid) and \
                       (np.abs(conn.controlState) == conn.controlPoints):
                        adj_watchtower = True

                    # See if player groups moving to the area
                    for gid in conn.groups[pid]:
                        in_group = self.players[pid].groups[gid]
                        if (in_group.moving == True) and \
                           (in_group.travel_destination == node.ID):
                               incoming_units = True
                               break
                    # end incoming group check


                if node.controlledBy == pid or stationed:
                    # full knowledge
                    knowledge[i] = 2
                    ctrl = node.controlledBy
                    pcnt = '{:.6f}'.format(100. * node.controlState / node.controlPoints)
                elif adj_watchtower or incoming_units:
                    # partial knowledge knowledge
                    knowledge[i] = 1
                    ctrl = node.controlledBy
                    pcnt = '{:.6f}'.format(100. * node.controlState / node.controlPoints)
                else:
                    # no change
                    ctrl = -1
                    pcnt = '{:.6f}'.format(0)
                nodes.append(node.ID)
                controller.append(ctrl)
                percent.append(pcnt)


            # end node knowledge loop

            # Node Knowledge Outstring
            outstr = '{:.6f},{},[{}],[{}],[{}],[{}]'.format(self.current_turn,
                                              pid,
                                              ';'.join(str(i) for i in nodes),
                                              ';'.join(str(i) for i in knowledge),
                                              ';'.join(str(i) for i in controller),
                                              ';'.join(str(i) for i in percent)
            )
            self.output['NODE_Knowledge'].append(outstr)

            # Group knowledge loop
            opp_k = {}  # player knowledge of enemy groups

            # Loop through nodes again now that we have knowledge of them all
            for i, nidx in enumerate(self.map_key2):
                nid = nodes[i]
                node = self.evgMap.nodes[nidx]
                if knowledge[i] == 1 or knowledge[i] == 2:
                    opp_k[nid] = {} # Dictionary of destinations

                    for opp_gid in node.groups[opp_pid]:
                        opp_group = self.players[opp_pid].groups[opp_gid]
                        in_ut = opp_group.units[0].unitType
                        ut = in_ut[0].upper() + in_ut[1:]
                        uc = opp_group.units[0].count
                        if opp_group.moving == False:
                            # Append as group staying put
                            if -1 in opp_k[nid]:
                                opp_k[nid][-1]['unitTypes'].append(ut)
                                opp_k[nid][-1]['unitCount'].append(uc)

                            else:
                                opp_k[nid][-1] = {'unitTypes':[ut],
                                                  'unitCount':[uc],
                                                  'status': 0
                                                 }
                            # end key existence check
                        else:
                            # Check for knowledge of destination
                            opp_dst = opp_group.travel_destination
                            dst_idx = nodes.index(opp_dst)
                            # Check knowledge of node id
                            if knowledge[dst_idx] > 0:
                                if dst_idx in opp_k[nid]:
                                    opp_k[nid][dst_idx]['unitTypes'].append(ut)
                                    opp_k[nid][dst_idx]['unitCount'].append(uc)

                                else:
                                    opp_k[nid][dst_idx] = {'unitTypes':[ut],
                                                           'unitCount':[uc],
                                                           'status': 0
                                                          }
                        # end group knowledge addition
                    # end group loop
                    if not bool(opp_k[nid]):
                        opp_k.pop(nid,None)
            # end group knowledge loop

            # Group knowledge outstring
            if bool(opp_k):
                for nid in opp_k.keys():
                    for dst in opp_k[nid].keys():
                        status = 0 if dst == -1 else 1
                        outstr = '{:.6f},{},[{}],[{}],{},{},{}'.format(
                                self.current_turn,
                                opp_pid,
                                ';'.join(str(i) for i in opp_k[nid][dst]['unitTypes']),
                                ';'.join(str(i) for i in opp_k[nid][dst]['unitCount']),
                                status,
                                nid,
                                dst
                        )
                        self.output['GROUP_Knowledge'].append(outstr)
                    # end destination key loop
                # end node key loop
            # end group knowledge outstring

        # end player loop

        
    def write_output(self):
        for key in self.output.keys():
            #pdb.set_trace()
            key_dir = self.dat_dir + '/' + str(key)
            oldmask = os.umask(000)
            os.mkdir(key_dir,mode=0o777)
            os.umask(oldmask)
            assert( os.path.isdir(key_dir) ), 'Could not create telemetry {} output directory'.format(key)

            key_file = key_dir + '/' + 'Telem_' + key
            with open(key_file, 'w') as fid:
                writer = csv.writer(fid, delimiter='\n')
                writer.writerow(self.output[key])



# end class EvergladesGame
