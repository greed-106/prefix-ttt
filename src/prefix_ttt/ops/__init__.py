"""Project-wide operator constants and the operator building blocks."""

# Fixed by the method, not by a run: the Prefix-TTT write scale (2**-7) and the
# Local-32 block size. Both are powers of two, so scaling by them is exact.
ETA = 0.0078125
LOCAL_BLOCK_SIZE = 32
TILE_SIZE = 64          # chunk size of the FLA prefix kernel

__all__ = ['ETA', 'LOCAL_BLOCK_SIZE', 'TILE_SIZE']
