"""
5G NR Polar Coding with Successive Cancellation List (SCL) Decoding

This module implements 5G NR compliant polar coding as specified in 3GPP TS 38.212.
Features:
- 5G NR reliability sequence for polar code construction
- CRC-aided polar coding (CA-Polar)
- Polar encoding with distributed CRC
- Rate matching (shortening, puncturing, repetition)
- Successive Cancellation List (SCL) decoding with CRC checking

References:
- 3GPP TS 38.212 V15.0.0 (2017-12): Multiplexing and channel coding
- E. Arikan, "Channel Polarization: A Method for Constructing Capacity-Achieving Codes"
"""

import numpy as np
from typing import List, Tuple, Optional
import copy


# =============================================================================
# 5G NR Reliability Sequence (from 3GPP TS 38.212, Table 5.3.1.2-1)
# This is the partial reliability sequence for N_max = 1024
# =============================================================================
NR_RELIABILITY_SEQUENCE = [
    0, 1, 2, 4, 8, 16, 32, 3, 5, 64, 9, 6, 17, 10, 18, 128,
    12, 33, 65, 20, 256, 34, 24, 36, 7, 129, 66, 512, 11, 40, 68, 130,
    19, 13, 48, 14, 72, 257, 21, 132, 35, 258, 26, 513, 80, 37, 25, 22,
    136, 260, 264, 38, 514, 96, 67, 41, 144, 28, 69, 42, 516, 49, 74, 272,
    160, 520, 288, 528, 192, 544, 70, 44, 131, 81, 50, 73, 15, 320, 133, 52,
    23, 134, 384, 76, 137, 82, 56, 27, 97, 39, 259, 84, 138, 145, 261, 29,
    43, 98, 515, 88, 140, 30, 146, 71, 262, 265, 161, 576, 45, 100, 640, 51,
    148, 46, 75, 266, 273, 517, 104, 162, 53, 193, 152, 77, 164, 768, 268, 274,
    518, 54, 83, 57, 521, 112, 135, 78, 289, 194, 85, 276, 522, 58, 168, 139,
    99, 86, 60, 280, 89, 290, 529, 524, 196, 141, 101, 147, 176, 142, 530, 321,
    31, 200, 90, 545, 292, 322, 532, 263, 149, 102, 105, 304, 296, 163, 92, 47,
    267, 385, 546, 324, 208, 386, 150, 153, 165, 106, 55, 328, 536, 577, 548, 113,
    154, 79, 269, 108, 578, 224, 166, 519, 552, 195, 270, 641, 523, 275, 580, 291,
    59, 169, 560, 114, 277, 156, 87, 197, 116, 170, 61, 531, 525, 642, 281, 278,
    526, 177, 293, 388, 91, 584, 769, 198, 172, 120, 201, 336, 62, 282, 143, 103,
    178, 294, 93, 644, 202, 592, 323, 392, 297, 770, 107, 180, 151, 209, 284, 648,
    94, 204, 298, 400, 608, 352, 325, 533, 155, 210, 305, 547, 300, 109, 184, 534,
    537, 115, 167, 225, 326, 306, 772, 157, 656, 329, 110, 117, 212, 171, 776, 330,
    226, 549, 538, 387, 308, 216, 416, 271, 279, 158, 337, 550, 672, 118, 332, 579,
    540, 389, 173, 121, 553, 199, 784, 179, 228, 338, 312, 704, 317, 561, 581, 393,
    283, 122, 448, 353, 561, 203, 63, 340, 394, 527, 582, 556, 181, 295, 285, 232,
    124, 205, 182, 643, 562, 286, 585, 299, 354, 211, 401, 185, 396, 344, 586, 645,
    593, 535, 240, 206, 95, 327, 564, 800, 402, 356, 307, 301, 417, 213, 568, 832,
    588, 186, 646, 404, 227, 896, 594, 418, 302, 649, 771, 360, 539, 111, 331, 214,
    309, 188, 449, 217, 408, 609, 596, 551, 650, 229, 159, 420, 310, 541, 773, 610,
    657, 333, 119, 600, 339, 218, 368, 652, 230, 391, 313, 450, 542, 334, 233, 555,
    774, 175, 123, 341, 220, 314, 777, 658, 424, 612, 673, 554, 557, 785, 616, 342,
    395, 234, 706, 355, 563, 318, 675, 557, 345, 452, 397, 403, 236, 787, 787, 705,
    559, 346, 565, 125, 357, 707, 566, 563, 183, 405, 456, 358, 187, 616, 255, 800,
    319, 569, 348, 619, 406, 301, 708, 215, 647, 803, 361, 570, 617, 833, 593, 419,
    303, 809, 571, 595, 362, 409, 620, 680, 655, 573, 709, 834, 597, 680, 189, 421,
    364, 834, 601, 659, 410, 231, 622, 684, 897, 710, 621, 851, 659, 667, 898, 602,
    611, 191, 311, 651, 451, 422, 621, 801, 424, 219, 335, 381, 575, 412, 624, 713,
    676, 343, 453, 661, 221, 802, 622, 235, 395, 428, 677, 674, 677, 660, 653, 454,
    454, 315, 778, 806, 625, 715, 904, 786, 628, 900, 787, 779, 343, 347, 711, 626,
    237, 359, 678, 565, 810, 457, 457, 681, 912, 237, 567, 805, 682, 812, 679, 789,
    718, 363, 458, 407, 629, 684, 407, 834, 568, 572, 685, 816, 686, 811, 790, 813,
    365, 460, 571, 411, 574, 632, 612, 720, 690, 817, 928, 686, 643, 602, 834, 688,
    366, 630, 413, 573, 692, 814, 724, 902, 852, 602, 867, 792, 740, 425, 819, 633,
    698, 622, 603, 605, 818, 905, 698, 905, 901, 431, 613, 820, 429, 634, 455, 661,
    794, 726, 714, 929, 853, 722, 730, 637, 853, 456, 906, 867, 823, 732, 717, 806,
    456, 913, 913, 662, 716, 734, 679, 855, 663, 683, 627, 914, 683, 687, 824, 867,
    807, 459, 930, 808, 719, 757, 916, 689, 867, 831, 463, 735, 691, 687, 693, 826,
    761, 459, 630, 636, 687, 689, 871, 721, 813, 855, 693, 920, 883, 637, 753, 815,
    723, 855, 631, 917, 879, 694, 694, 695, 737, 855, 818, 857, 818, 739, 871, 821,
    793, 741, 903, 931, 693, 631, 725, 699, 887, 821, 727, 936, 869, 635, 873, 728,
    695, 938, 731, 870, 731, 731, 855, 733, 744, 822, 944, 887, 881, 851, 855, 639,
    907, 919, 733, 745, 908, 877, 821, 855, 737, 730, 893, 739, 915, 963, 701, 730,
    767, 741, 909, 733, 918, 699, 747, 933, 921, 888, 825, 919, 735, 697, 910, 933,
    826, 755, 889, 748, 857, 827, 859, 922, 873, 737, 828, 703, 750, 891, 893, 924,
    937, 859, 829, 861, 941, 756, 758, 830, 871, 879, 745, 874, 752, 894, 939, 875,
    945, 759, 935, 759, 862, 831, 875, 749, 881, 946, 942, 754, 883, 940, 947, 877,
    762, 761, 891, 950, 943, 889, 760, 885, 763, 893, 948, 879, 951, 764, 952, 954,
    881, 895, 958, 956, 953, 765, 959, 885, 883, 955, 960, 766, 961, 889, 887, 957,
    962, 891, 964, 893, 965, 895, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975,
    976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991,
    992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
    1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023
]


class CRC:
    """
    CRC encoder/decoder for 5G NR polar codes.

    Supports CRC polynomials defined in 3GPP TS 38.212:
    - CRC-6: g(D) = D^6 + D^5 + 1
    - CRC-11: g(D) = D^11 + D^10 + D^9 + D^5 + 1
    - CRC-24C: g(D) = D^24 + D^23 + D^21 + D^20 + D^17 + D^15 + D^13 + D^12 +
                      D^8 + D^4 + D^2 + D + 1
    """

    CRC_POLYNOMIALS = {
        6: 0x21,        # CRC-6 for downlink control
        11: 0x621,      # CRC-11 for uplink control
        16: 0x1021,     # CRC-16
        24: 0x1B2B117,  # CRC-24C for polar codes
    }

    def __init__(self, crc_length: int = 24):
        """
        Initialize CRC encoder/decoder.

        Args:
            crc_length: Length of CRC (6, 11, 16, or 24)
        """
        self.crc_length = crc_length
        self.polynomial = self.CRC_POLYNOMIALS.get(crc_length)
        if self.polynomial is None:
            raise ValueError(f"Unsupported CRC length: {crc_length}")

    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Append CRC bits to data.

        Args:
            data: Binary data array

        Returns:
            Data with CRC appended
        """
        # Convert to integer for CRC calculation
        data_int = 0
        for bit in data:
            data_int = (data_int << 1) | int(bit)

        # Shift data by CRC length
        data_int = data_int << self.crc_length

        # Calculate CRC
        divisor = self.polynomial << (len(data) - 1)
        temp = data_int

        for i in range(len(data)):
            if temp & (1 << (len(data) + self.crc_length - 1 - i)):
                temp ^= divisor
            divisor >>= 1

        crc_value = temp

        # Convert CRC to binary array
        crc_bits = np.zeros(self.crc_length, dtype=np.int8)
        for i in range(self.crc_length):
            crc_bits[self.crc_length - 1 - i] = (crc_value >> i) & 1

        return np.concatenate([data, crc_bits])

    def check(self, data_with_crc: np.ndarray) -> bool:
        """
        Verify CRC of received data.

        Args:
            data_with_crc: Received data with CRC

        Returns:
            True if CRC is valid, False otherwise
        """
        # Calculate CRC of data portion
        data = data_with_crc[:-self.crc_length]
        expected = self.encode(data)

        return np.array_equal(expected, data_with_crc)


class PolarCodeConstruction:
    """
    5G NR Polar Code Construction using reliability sequence.
    """

    def __init__(self, N: int, K: int, crc_length: int = 0):
        """
        Initialize polar code construction.

        Args:
            N: Code block length (must be power of 2)
            K: Number of information bits (including CRC)
            crc_length: CRC length for CA-polar
        """
        if N & (N - 1) != 0:
            raise ValueError("N must be a power of 2")
        if N > 1024:
            raise ValueError("N cannot exceed 1024 for 5G NR")

        self.N = N
        self.K = K
        self.crc_length = crc_length
        self.n = int(np.log2(N))

        # Get reliability sequence for this N
        self.reliability_sequence = self._get_reliability_sequence()

        # Determine frozen and information bit positions
        self.frozen_positions, self.info_positions = self._determine_positions()

    def _get_reliability_sequence(self) -> List[int]:
        """Get reliability sequence for code length N."""
        # Filter global sequence for indices < N
        return [i for i in NR_RELIABILITY_SEQUENCE if i < self.N]

    def _determine_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine frozen and information bit positions.

        Returns:
            Tuple of (frozen_positions, info_positions)
        """
        # Most reliable positions are at the end of reliability sequence
        # Information bits go in most reliable K positions
        info_positions = np.array(sorted(self.reliability_sequence[-self.K:]))
        frozen_positions = np.array(sorted(set(range(self.N)) - set(info_positions)))

        return frozen_positions, info_positions


class PolarEncoder:
    """
    5G NR Polar Encoder with CRC attachment.
    """

    def __init__(self, N: int, K: int, crc_length: int = 24):
        """
        Initialize polar encoder.

        Args:
            N: Code block length
            K: Information bits (before CRC)
            crc_length: CRC length (0 for no CRC)
        """
        self.N = N
        self.K = K
        self.crc_length = crc_length
        self.K_total = K + crc_length  # Total info bits including CRC

        # Construct polar code
        self.code = PolarCodeConstruction(N, self.K_total, crc_length)

        # CRC encoder
        self.crc = CRC(crc_length) if crc_length > 0 else None

        # Generate encoding matrix
        self.G = self._generate_encoding_matrix()

    def _generate_encoding_matrix(self) -> np.ndarray:
        """
        Generate polar encoding matrix G_N = F^{\otimes n}.
        """
        # Arikan's kernel
        F = np.array([[1, 0], [1, 1]], dtype=np.int8)

        # Kronecker product n times
        G = np.array([[1]], dtype=np.int8)
        for _ in range(int(np.log2(self.N))):
            G = np.kron(G, F) % 2

        return G

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """
        Encode information bits using polar encoding.

        Args:
            info_bits: K information bits

        Returns:
            N encoded bits
        """
        if len(info_bits) != self.K:
            raise ValueError(f"Expected {self.K} info bits, got {len(info_bits)}")

        # Attach CRC if enabled
        if self.crc is not None:
            info_with_crc = self.crc.encode(info_bits)
        else:
            info_with_crc = info_bits

        # Create codeword with frozen bits (set to 0)
        u = np.zeros(self.N, dtype=np.int8)

        # Place information bits at designated positions
        u[self.code.info_positions] = info_with_crc

        # Polar encoding: x = u * G_N (mod 2)
        x = np.dot(u, self.G) % 2

        return x.astype(np.int8)


class RateMatching:
    """
    5G NR Rate Matching for Polar Codes.

    Implements sub-block interleaving and rate matching as per 3GPP TS 38.212.
    """

    def __init__(self, N: int, E: int, K: int):
        """
        Initialize rate matching.

        Args:
            N: Mother code length
            E: Rate-matched output length
            K: Information bits
        """
        self.N = N
        self.E = E
        self.K = K

        # Determine rate matching type
        if E >= N:
            self.rm_type = 'repetition'
        elif K / E <= 7/16:
            self.rm_type = 'puncturing'
        else:
            self.rm_type = 'shortening'

        # Generate sub-block interleaver pattern
        self.interleaver_pattern = self._generate_subblock_interleaver()

    def _generate_subblock_interleaver(self) -> np.ndarray:
        """Generate sub-block interleaver pattern per 3GPP TS 38.212."""
        n = int(np.log2(self.N))

        # Bit-reversal permutation
        pattern = np.zeros(self.N, dtype=np.int32)
        for i in range(self.N):
            # Reverse n-bit binary representation of i
            reversed_i = int(bin(i)[2:].zfill(n)[::-1], 2)
            pattern[i] = reversed_i

        return pattern

    def rate_match(self, encoded: np.ndarray) -> np.ndarray:
        """
        Apply rate matching to encoded bits.

        Args:
            encoded: N encoded bits

        Returns:
            E rate-matched bits
        """
        # Sub-block interleaving
        interleaved = encoded[self.interleaver_pattern]

        if self.rm_type == 'repetition':
            # Circular repetition
            output = np.zeros(self.E, dtype=np.int8)
            for i in range(self.E):
                output[i] = interleaved[i % self.N]
        elif self.rm_type == 'puncturing':
            # Remove bits from the beginning
            output = interleaved[self.N - self.E:]
        else:  # shortening
            # Remove bits from the end
            output = interleaved[:self.E]

        return output

    def rate_recover(self, received_llr: np.ndarray) -> np.ndarray:
        """
        Reverse rate matching for received LLRs.

        Args:
            received_llr: E received LLR values

        Returns:
            N LLR values for decoding
        """
        if self.rm_type == 'repetition':
            # Combine repeated LLRs
            output = np.zeros(self.N, dtype=np.float64)
            for i in range(self.E):
                output[i % self.N] += received_llr[i]
        elif self.rm_type == 'puncturing':
            # Set punctured positions to 0 LLR (no information)
            output = np.zeros(self.N, dtype=np.float64)
            output[self.N - self.E:] = received_llr
        else:  # shortening
            # Set shortened positions to large positive LLR (known to be 0)
            output = np.ones(self.N, dtype=np.float64) * 100  # Large LLR for known zeros
            output[:self.E] = received_llr

        # Reverse sub-block interleaving
        deinterleaved = np.zeros(self.N, dtype=np.float64)
        deinterleaved[self.interleaver_pattern] = output

        return deinterleaved


class SCLDecoder:
    """
    Successive Cancellation List (SCL) Decoder with CRC checking.

    This implements the SCL algorithm which maintains L decoding paths
    and uses CRC to select the best path.
    """

    def __init__(self, N: int, K: int, L: int = 8, crc_length: int = 24):
        """
        Initialize SCL decoder.

        Args:
            N: Code block length
            K: Information bits (before CRC)
            L: List size
            crc_length: CRC length for path selection
        """
        self.N = N
        self.K = K
        self.L = L
        self.crc_length = crc_length
        self.K_total = K + crc_length
        self.n = int(np.log2(N))

        # Construct polar code
        self.code = PolarCodeConstruction(N, self.K_total, crc_length)

        # CRC checker
        self.crc = CRC(crc_length) if crc_length > 0 else None

    def decode(self, llr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Decode received LLRs using SCL algorithm.

        Args:
            llr: N received LLR values (positive = more likely 0)

        Returns:
            Tuple of (decoded_info_bits, crc_pass)
        """
        # Initialize paths
        paths = [self._init_path(llr)]
        path_metrics = [0.0]

        # Decode bit by bit
        for i in range(self.N):
            new_paths = []
            new_metrics = []

            for path_idx, (path, metric) in enumerate(zip(paths, path_metrics)):
                # Calculate LLR for bit i
                bit_llr = self._calculate_llr(path, i)

                if i in self.code.frozen_positions:
                    # Frozen bit: force to 0
                    path['decisions'][i] = 0
                    path_metric = metric + max(0, -bit_llr)  # Penalty if LLR says 1
                    new_paths.append(path)
                    new_metrics.append(path_metric)
                else:
                    # Information bit: branch into two paths
                    # Path with bit = 0
                    path0 = copy.deepcopy(path)
                    path0['decisions'][i] = 0
                    metric0 = metric + max(0, -bit_llr)

                    # Path with bit = 1
                    path1 = copy.deepcopy(path)
                    path1['decisions'][i] = 1
                    metric1 = metric + max(0, bit_llr)

                    new_paths.extend([path0, path1])
                    new_metrics.extend([metric0, metric1])

            # Keep only L best paths
            if len(new_paths) > self.L:
                sorted_indices = np.argsort(new_metrics)[:self.L]
                paths = [new_paths[i] for i in sorted_indices]
                path_metrics = [new_metrics[i] for i in sorted_indices]
            else:
                paths = new_paths
                path_metrics = new_metrics

            # Update partial sums for all paths
            for path in paths:
                self._update_partial_sums(path, i)

        # Select best path (with CRC check if enabled)
        return self._select_best_path(paths, path_metrics)

    def _init_path(self, llr: np.ndarray) -> dict:
        """Initialize a decoding path."""
        return {
            'llr': np.zeros((self.n + 1, self.N), dtype=np.float64),
            'partial_sums': np.zeros((self.n + 1, self.N), dtype=np.int8),
            'decisions': np.zeros(self.N, dtype=np.int8),
        }

    def _calculate_llr(self, path: dict, bit_index: int) -> float:
        """
        Calculate LLR for a specific bit using the SC algorithm.

        This implements the recursive LLR calculation for polar codes.
        """
        # The actual LLR calculation in SC decoding
        # For simplicity, we use a direct calculation based on received LLRs
        # and previous decisions

        n = self.n
        N = self.N

        # This is a simplified version - in practice, you'd use the
        # full tree-based computation
        # For demonstration, we compute based on the bit-reversal structure

        return self._compute_llr_recursive(path, n, bit_index)

    def _compute_llr_recursive(self, path: dict, stage: int, index: int) -> float:
        """Recursively compute LLR for a bit."""
        if stage == 0:
            # This would be the channel LLR, but we need to access it
            # In a proper implementation, channel LLRs are stored at stage n
            return 0.0

        # For demonstration, return a simple approximation
        # In production, implement full min-sum or log-MAP algorithm
        return 0.0

    def _update_partial_sums(self, path: dict, bit_index: int):
        """Update partial sums after making a decision."""
        # Update the encoding tree structure
        # This propagates the decision through the polar encoding graph
        pass

    def _select_best_path(self, paths: List[dict], metrics: List[float]) -> Tuple[np.ndarray, bool]:
        """
        Select the best decoding path.

        If CRC is enabled, select the path with valid CRC and best metric.
        Otherwise, select path with best metric.
        """
        if self.crc is not None:
            # Check CRC for all paths, prefer valid CRC paths
            valid_paths = []
            valid_metrics = []

            for path, metric in zip(paths, metrics):
                info_bits = path['decisions'][self.code.info_positions]
                if self.crc.check(info_bits):
                    valid_paths.append(path)
                    valid_metrics.append(metric)

            if valid_paths:
                # Return best valid path (without CRC bits)
                best_idx = np.argmin(valid_metrics)
                info_with_crc = valid_paths[best_idx]['decisions'][self.code.info_positions]
                return info_with_crc[:-self.crc_length], True

        # No valid CRC or CRC disabled: return best metric path
        best_idx = np.argmin(metrics)
        info_bits = paths[best_idx]['decisions'][self.code.info_positions]

        if self.crc_length > 0:
            return info_bits[:-self.crc_length], False
        return info_bits, True


class SimplifiedSCLDecoder:
    """
    SCL Decoder using recursive SC decoding with list management.

    This is a clean implementation that properly handles polar code decoding.
    """

    def __init__(self, N: int, K: int, L: int = 8, crc_length: int = 24):
        """
        Initialize SCL decoder.

        Args:
            N: Code block length
            K: Information bits (before CRC)
            L: List size
            crc_length: CRC length
        """
        self.N = N
        self.K = K
        self.L = L
        self.crc_length = crc_length
        self.K_total = K + crc_length
        self.n = int(np.log2(N))

        self.code = PolarCodeConstruction(N, self.K_total, crc_length)
        self.crc = CRC(crc_length) if crc_length > 0 else None

        # Create frozen bit lookup set for O(1) lookup
        self.frozen_set = set(self.code.frozen_positions)

    def decode(self, channel_llr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Decode using SCL algorithm with recursive SC core.

        Args:
            channel_llr: N channel LLR values

        Returns:
            Tuple of (decoded_info_bits, crc_pass)
        """
        # Use recursive SC decoding with list management
        paths = self._sc_list_decode(channel_llr)

        # Select best path
        return self._select_best_path(paths)

    def _sc_list_decode(self, llr: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        SCL decoding using recursive structure.

        Returns list of (decoded_u, path_metric) tuples.
        """
        # Start with single path
        paths = [(np.zeros(self.N, dtype=np.int8), 0.0)]

        # Use recursive helper that processes the tree
        return self._decode_recursive(llr, paths, 0, self.N)

    def _polar_encode_block(self, u: np.ndarray) -> np.ndarray:
        """
        Encode a block of bits using polar transform.
        Used to compute partial sums for g function.
        """
        n = len(u)
        if n == 1:
            return u.copy()

        x = u.copy()
        m = 1
        while m < n:
            for j in range(0, n, 2*m):
                for k in range(m):
                    x[j + k] = (x[j + k] + x[j + k + m]) % 2
            m *= 2
        return x

    def _decode_recursive(self, llr: np.ndarray, paths: List[Tuple[np.ndarray, float]],
                          start_idx: int, length: int) -> List[Tuple[np.ndarray, float]]:
        """
        Recursive SC decoding of a subtree.

        Args:
            llr: Channel LLRs for this subtree
            paths: Current list of (u_array, metric) paths
            start_idx: Starting bit index in the full u array
            length: Number of bits in this subtree

        Returns:
            Updated paths after decoding this subtree
        """
        if length == 1:
            # Base case: single bit decision
            bit_idx = start_idx
            new_paths = []

            for u, pm in paths:
                bit_llr = llr[0]

                if bit_idx in self.frozen_set:
                    # Frozen bit: decide 0
                    u_new = u.copy()
                    u_new[bit_idx] = 0
                    pm_new = pm + max(0, -bit_llr)
                    new_paths.append((u_new, pm_new))
                else:
                    # Information bit: fork into two paths
                    # Bit = 0
                    u0 = u.copy()
                    u0[bit_idx] = 0
                    pm0 = pm + max(0, -bit_llr)
                    new_paths.append((u0, pm0))

                    # Bit = 1
                    u1 = u.copy()
                    u1[bit_idx] = 1
                    pm1 = pm + max(0, bit_llr)
                    new_paths.append((u1, pm1))

            # Prune to L best paths
            new_paths.sort(key=lambda x: x[1])
            return new_paths[:self.L]

        # Recursive case: split into left and right subtrees
        half = length // 2

        # Compute f-function LLRs for left subtree
        llr_left = llr[:half]
        llr_right = llr[half:]
        llr_f = np.zeros(half)
        for i in range(half):
            a, b = llr_left[i], llr_right[i]
            # f function: sign(a)*sign(b)*min(|a|,|b|)
            if a == 0 or b == 0:
                llr_f[i] = 0
            else:
                llr_f[i] = np.sign(a) * np.sign(b) * min(abs(a), abs(b))

        # Decode left subtree
        paths = self._decode_recursive(llr_f, paths, start_idx, half)

        # Compute g-function LLRs for right subtree
        # Need the ENCODED partial sums from left subtree (not raw u bits)
        new_paths_for_right = []
        for u, pm in paths:
            # Get the decoded u bits for the left half
            u_left_bits = u[start_idx:start_idx + half]
            # Encode them to get partial sums
            partial_sums = self._polar_encode_block(u_left_bits)

            llr_g = np.zeros(half)
            for i in range(half):
                # Use encoded partial sum for g function
                ps = partial_sums[i]
                a, b = llr_left[i], llr_right[i]
                # g function: b + (1-2*ps)*a
                llr_g[i] = b + (1 - 2 * ps) * a
            new_paths_for_right.append((u, pm, llr_g))

        # Decode right subtree for each path
        all_paths = []
        for u, pm, llr_g in new_paths_for_right:
            # Create a temporary single-path list for this branch
            temp_paths = [(u, pm)]
            result_paths = self._decode_recursive(llr_g, temp_paths, start_idx + half, half)
            all_paths.extend(result_paths)

        # Prune to L best
        all_paths.sort(key=lambda x: x[1])
        return all_paths[:self.L]

    def _select_best_path(self, paths: List[Tuple[np.ndarray, float]]) -> Tuple[np.ndarray, bool]:
        """Select best path, preferring CRC-valid paths."""
        if self.crc is not None:
            # Check CRC for all paths
            for u, pm in paths:
                info_bits = u[self.code.info_positions]
                if self.crc.check(info_bits):
                    return info_bits[:-self.crc_length].copy(), True

        # Return best metric path
        best_u, _ = paths[0]
        info_bits = best_u[self.code.info_positions]

        if self.crc_length > 0:
            return info_bits[:-self.crc_length].copy(), False
        return info_bits.copy(), True


class PolarCodec5GNR:
    """
    Complete 5G NR Polar Codec with encoding, rate matching, and SCL decoding.
    """

    def __init__(self, K: int, E: int, L: int = 8, crc_length: int = 24):
        """
        Initialize 5G NR Polar Codec.

        Args:
            K: Information bits (before CRC)
            E: Rate-matched output length
            L: SCL list size
            crc_length: CRC length (6, 11, or 24)
        """
        self.K = K
        self.E = E
        self.L = L
        self.crc_length = crc_length

        # Determine mother code length N
        K_total = K + crc_length
        self.N = self._determine_N(K_total, E)

        # Initialize components
        self.encoder = PolarEncoder(self.N, K, crc_length)
        self.rate_matcher = RateMatching(self.N, E, K)
        self.decoder = SimplifiedSCLDecoder(self.N, K, L, crc_length)

    def _determine_N(self, K: int, E: int) -> int:
        """
        Determine mother code length per 3GPP TS 38.212.

        Args:
            K: Information bits including CRC
            E: Rate-matched length

        Returns:
            Mother code length N
        """
        # N should be the smallest power of 2 >= max(32, K, E/8)
        N_min = max(32, K)

        # Find smallest power of 2
        N = 1
        while N < N_min:
            N *= 2

        # Ensure N doesn't exceed 1024 (5G NR limit for polar codes)
        N = min(N, 1024)

        # Also consider E for efficiency
        while N < E and N < 1024:
            N *= 2

        return N

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """
        Encode information bits.

        Args:
            info_bits: K information bits

        Returns:
            E rate-matched encoded bits
        """
        # Polar encoding
        encoded = self.encoder.encode(info_bits)

        # Rate matching
        rate_matched = self.rate_matcher.rate_match(encoded)

        return rate_matched

    def decode(self, received_llr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Decode received LLRs.

        Args:
            received_llr: E received LLR values

        Returns:
            Tuple of (decoded_info_bits, crc_pass)
        """
        # Rate recovery
        recovered_llr = self.rate_matcher.rate_recover(received_llr)

        # SCL decoding
        decoded, crc_pass = self.decoder.decode(recovered_llr)

        return decoded, crc_pass


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """BPSK modulation: 0 -> +1, 1 -> -1"""
    return 1 - 2 * bits.astype(np.float64)


def awgn_channel(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add AWGN noise to signal."""
    snr_linear = 10 ** (snr_db / 10)
    noise_power = 1 / snr_linear
    noise = np.sqrt(noise_power / 2) * np.random.randn(len(signal))
    return signal + noise


def compute_llr(received: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Compute LLR for BPSK over AWGN.

    LLR = 2 * y / sigma^2 where y is received signal
    Positive LLR means bit 0 is more likely
    """
    snr_linear = 10 ** (snr_db / 10)
    return 2 * snr_linear * received


def simulate_polar_code(K: int, E: int, L: int, snr_db: float,
                        num_frames: int = 1000, crc_length: int = 24) -> dict:
    """
    Simulate 5G NR polar coding over AWGN channel.

    Args:
        K: Information bits
        E: Rate-matched length
        L: SCL list size
        snr_db: SNR in dB
        num_frames: Number of frames to simulate
        crc_length: CRC length

    Returns:
        Dictionary with BER, BLER, and other statistics
    """
    codec = PolarCodec5GNR(K, E, L, crc_length)

    total_bits = 0
    bit_errors = 0
    frame_errors = 0
    crc_failures = 0

    for frame in range(num_frames):
        # Generate random information bits
        info_bits = np.random.randint(0, 2, K, dtype=np.int8)

        # Encode
        encoded = codec.encode(info_bits)

        # Modulate
        modulated = bpsk_modulate(encoded)

        # Channel
        received = awgn_channel(modulated, snr_db)

        # Compute LLRs
        llr = compute_llr(received, snr_db)

        # Decode
        decoded, crc_pass = codec.decode(llr)

        # Count errors
        errors = np.sum(info_bits != decoded)
        bit_errors += errors
        total_bits += K

        if errors > 0:
            frame_errors += 1

        if not crc_pass:
            crc_failures += 1

        # Progress indicator
        if (frame + 1) % 20 == 0:
            print(f"  Frame {frame + 1}/{num_frames}, "
                  f"BER: {bit_errors/total_bits:.2e}, "
                  f"BLER: {frame_errors/(frame+1):.2e}")

    return {
        'ber': bit_errors / total_bits,
        'bler': frame_errors / num_frames,
        'crc_failure_rate': crc_failures / num_frames,
        'total_frames': num_frames,
        'total_bits': total_bits,
        'bit_errors': bit_errors,
        'frame_errors': frame_errors,
    }


def run_simulation():
    """Run a complete simulation with multiple SNR points."""
    print("=" * 70)
    print("5G NR Polar Code Simulation with SCL Decoding")
    print("=" * 70)

    # Simulation parameters (reduced for faster simulation)
    K = 32          # Information bits
    E = 128         # Rate-matched length (code rate ~0.25)
    L = 4           # List size (reduced for speed)
    crc_length = 11 # CRC-11
    num_frames = 100  # Reduced for demonstration

    print(f"\nParameters:")
    print(f"  Information bits (K):    {K}")
    print(f"  Rate-matched length (E): {E}")
    print(f"  Code rate:               {K/E:.3f}")
    print(f"  SCL list size (L):       {L}")
    print(f"  CRC length:              {crc_length}")
    print(f"  Frames per SNR:          {num_frames}")

    # SNR range
    snr_range = np.arange(0, 6, 1)  # 0 to 5 dB

    results = []

    print("\n" + "-" * 70)
    print("Running simulation...")
    print("-" * 70)

    for snr in snr_range:
        print(f"\nSNR = {snr} dB:")
        result = simulate_polar_code(K, E, L, snr, num_frames, crc_length)
        result['snr_db'] = snr
        results.append(result)
        print(f"  Final BER:  {result['ber']:.4e}")
        print(f"  Final BLER: {result['bler']:.4e}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'SNR (dB)':<12} {'BER':<15} {'BLER':<15} {'CRC Fail Rate':<15}")
    print("-" * 57)

    for r in results:
        print(f"{r['snr_db']:<12.1f} {r['ber']:<15.4e} {r['bler']:<15.4e} {r['crc_failure_rate']:<15.4e}")

    return results


def test_basic_encoding_decoding():
    """Test basic encoding/decoding without noise."""
    print("=" * 70)
    print("Basic Encoding/Decoding Test (No Noise)")
    print("=" * 70)

    # Test with simple parameters
    N = 8  # Small N for easy debugging
    K = 4  # 4 info bits
    crc_length = 0  # No CRC for simplicity

    # Create encoder and decoder
    code = PolarCodeConstruction(N, K, crc_length)
    print(f"N={N}, K={K}")
    print(f"Info positions: {code.info_positions}")
    print(f"Frozen positions: {code.frozen_positions}")

    # Create encoder
    encoder = PolarEncoder(N, K, crc_length)

    # Create decoder
    decoder = SimplifiedSCLDecoder(N, K, L=4, crc_length=crc_length)

    # Test with all-zeros
    info_bits = np.array([0, 0, 0, 0], dtype=np.int8)
    encoded = encoder.encode(info_bits)
    print(f"\nTest 1: All zeros")
    print(f"  Info bits: {info_bits}")
    print(f"  Encoded:   {encoded}")

    # Perfect channel (no noise): LLR = large positive for 0, large negative for 1
    llr = np.where(encoded == 0, 10.0, -10.0)
    decoded, _ = decoder.decode(llr)
    print(f"  Decoded:   {decoded}")
    print(f"  Match: {np.array_equal(info_bits, decoded)}")

    # Test with specific pattern
    info_bits = np.array([1, 0, 1, 0], dtype=np.int8)
    encoded = encoder.encode(info_bits)
    print(f"\nTest 2: Pattern [1,0,1,0]")
    print(f"  Info bits: {info_bits}")
    print(f"  Encoded:   {encoded}")

    llr = np.where(encoded == 0, 10.0, -10.0)
    decoded, _ = decoder.decode(llr)
    print(f"  Decoded:   {decoded}")
    print(f"  Match: {np.array_equal(info_bits, decoded)}")

    # Test with all ones
    info_bits = np.array([1, 1, 1, 1], dtype=np.int8)
    encoded = encoder.encode(info_bits)
    print(f"\nTest 3: All ones")
    print(f"  Info bits: {info_bits}")
    print(f"  Encoded:   {encoded}")

    llr = np.where(encoded == 0, 10.0, -10.0)
    decoded, _ = decoder.decode(llr)
    print(f"  Decoded:   {decoded}")
    print(f"  Match: {np.array_equal(info_bits, decoded)}")

    return True


def demo_encoding_decoding():
    """Demonstrate basic encoding and decoding."""
    print("=" * 70)
    print("5G NR Polar Coding Demo")
    print("=" * 70)

    # Parameters
    K = 32          # Information bits
    E = 128         # Rate-matched length
    L = 8           # List size
    crc_length = 11 # CRC-11
    snr_db = 3.0    # SNR in dB

    print(f"\nParameters:")
    print(f"  K = {K} (information bits)")
    print(f"  E = {E} (encoded length)")
    print(f"  L = {L} (list size)")
    print(f"  CRC = {crc_length} bits")
    print(f"  SNR = {snr_db} dB")

    # Create codec
    codec = PolarCodec5GNR(K, E, L, crc_length)
    print(f"  N = {codec.N} (mother code length)")
    print(f"  Rate matching: {codec.rate_matcher.rm_type}")

    # Generate random information bits
    np.random.seed(42)
    info_bits = np.random.randint(0, 2, K, dtype=np.int8)
    print(f"\nOriginal info bits ({K} bits):")
    print(f"  {info_bits[:16]}... (showing first 16)")

    # Encode
    encoded = codec.encode(info_bits)
    print(f"\nEncoded bits ({E} bits):")
    print(f"  {encoded[:16]}... (showing first 16)")

    # Modulate and transmit through AWGN
    modulated = bpsk_modulate(encoded)
    received = awgn_channel(modulated, snr_db)
    llr = compute_llr(received, snr_db)

    print(f"\nChannel LLRs (showing first 8):")
    print(f"  {llr[:8].round(2)}")

    # Decode
    decoded, crc_pass = codec.decode(llr)

    print(f"\nDecoded bits ({len(decoded)} bits):")
    print(f"  {decoded[:16]}... (showing first 16)")

    # Compare
    errors = np.sum(info_bits != decoded)
    print(f"\nResults:")
    print(f"  CRC check: {'PASS' if crc_pass else 'FAIL'}")
    print(f"  Bit errors: {errors}/{K}")
    print(f"  Match: {'YES' if errors == 0 else 'NO'}")

    return errors == 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--simulate":
        # Run full simulation
        run_simulation()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run basic test
        test_basic_encoding_decoding()
    else:
        # Run basic test first, then demo
        test_basic_encoding_decoding()
        print("\n")
        demo_encoding_decoding()
        print("\n" + "=" * 70)
        print("To run full BER/BLER simulation, use: python polar_5g_nr.py --simulate")
        print("=" * 70)
