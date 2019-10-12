import time
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

import rubiks2
from superflip import superflip_set


def maxDistance(arr):

    # Used to store element to first index mapping
    mp = {}

    # Traverse elements and find maximum distance between
    # same occurrences with the help of map.
    maxDict = 0
    maxFirst_idx = 0
    for i in range(len(arr)):

        # If this is first occurrence of element, insert its
        # index in map
        if arr[i] not in mp.keys():
            mp[arr[i]] = i

        # Else update max distance
        else:
            if i-mp[arr[i]]>maxDict:
                maxDict = max(maxDict, i-mp[arr[i]])
                maxFirst_idx = mp[arr[i]]

    return maxDict, maxFirst_idx

def test_rubiks(network, device, max_tries=None):
    solve_rate_superflip = np.zeros(14)
    counts_superflip = np.zeros(14)
    seq_len_superflip = np.zeros(14)
    seq_len_superflip_heur = np.zeros(14)
    puzzles = []
    solution_sequences = []

    try:
        for i in range(14):
            hashes_seqs = []

            # If the previous distance could not be solved then it makes no sense trying the next
            if i > 0:
                if solve_rate_superflip[i-1] == 0:
                    break

            for sequence in tqdm(superflip_set):
                env = rubiks2.RubiksEnv2(2, unsolved_reward=-1.0)

                hashed_sequence = hash(str(sequence[:i+1]))

                if hashed_sequence not in hashes_seqs:

                    hashes_seqs.append(hashed_sequence)

                    counts_superflip[i] += 1

                    puzzle = []
                    for j in range(i + 1):
                        env.step(int(sequence[j]))
                        puzzle.append(sequence[j])

                    puzzles.append(puzzle)

                    hashes = defaultdict(list)
                    done = 0
                    tries = 0
                    t = time.time()
                    state = env.get_observation()
                    hashes[hash(state.tostring())] = [0]*env.action_space.n
                    stop = False

                    solution_sequence = []
                    state_hash_seq = []
                    while time.time()-t < 1.21 and not done and not stop:
                        mask = hashes[hash(state.tostring())]
                        state_hash_seq.append(hash(state.tostring()))
                        action = network.act(state, 0.0, mask, device)
                        solution_sequence.append(action)

                        next_state, reward, done, info = env.step(action)

                        hstate = state.copy()
                        state = next_state
                        h = hash(state.tostring())
                        if h in hashes.keys():
                            hashes[hash(hstate.tostring())][action] = -999
                        else:
                            hashes[h] = [0]*env.action_space.n

                        tries += 1
                        if max_tries:
                            if tries >= max_tries:
                                stop = True

                    length, first_idx = maxDistance(state_hash_seq)

                    # Remove redundant steps in the sequence
                    while length > 0:
                        state_hash_seq = state_hash_seq[:first_idx] + state_hash_seq[first_idx + length:]
                        solution_sequence = solution_sequence[:first_idx] + solution_sequence[first_idx + length:]
                        length, first_idx = maxDistance(state_hash_seq)

                    solution_sequences.append(solution_sequence)
                    solve_rate_superflip[i] += done

                    if done:
                        seq_len_superflip[i] += tries
                        seq_len_superflip_heur[i] += len(solution_sequence)

            print(solve_rate_superflip[i]/counts_superflip[i])
    except KeyboardInterrupt:
        pass

    score = np.zeros(14)
    solve_rate = np.divide(solve_rate_superflip, counts_superflip)
    seq_len = np.divide(seq_len_superflip, solve_rate_superflip)
    for i in range(14):
        score[i] = solve_rate[i] / (1+(seq_len[i] - (i+1)))

    score = np.mean(score) - np.std(score)

    return (np.mean(np.divide(solve_rate_superflip, counts_superflip))-np.std(np.divide(solve_rate_superflip, counts_superflip)),
            score,
            np.divide(solve_rate_superflip, counts_superflip),
np.divide(seq_len_superflip, solve_rate_superflip),
            np.divide(seq_len_superflip_heur, solve_rate_superflip),
           puzzles, solution_sequences)

if __name__ == "__main__":
    res = test_rubiks(None, 'cpu', 1000)
    print(res[2])
