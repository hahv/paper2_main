+----------------------+-------+---------+-----------------------------------------------------------+-------------------------------------------------+
|                      |       | OVERALL |                DETAILED DECISION BREAKDOWN                |             VISUALIZATION (Timeline)            |
| VIDEO NAME           | FRAMES| SKIP %  |  Miss (FN)    |  Waste (FP)  | True Skip (TN) | Hit (TP) | [Top: GT]  /  [Bottom: PRED]                  |
    +----------------------+-------+---------+---------------+--------------+----------------+----------+-------------------------------------------------+
|                      |       |         |               |              |                |          | GT : [Safe...........][FIRE!!!!!!!!][Safe.....] |
| Scenario_A_Ideal.mp4 |  500  |   90%   |      0%       |      2%      |      88%       |    10%   | PRED: [Skipped........][Processed...][Skipped..] |
|                      |       |         |    (Safe)     |              |                |          |       (Perfect Match)                           |
+----------------------+-------+---------+---------------+--------------+----------------+----------+-------------------------------------------------+
|                      |       |         |               |              |                |          | GT : [Safe...........][FIRE!!!!!!!!][Safe.....] |
| Scenario_B_Miss.mp4  |  500  |   95%   |     20%       |      0%      |      75%       |     5%   | ACT: [Skipped........][Skipped.....][Skipped..] |
|                      |       |         |  (DANGEROUS)  |              |                |          |                       ^ DANGER (Missed Fire)    |
+----------------------+-------+---------+---------------+--------------+----------------+----------+-------------------------------------------------+
|                      |       |         |               |              |                |          | GT : [Safe...........][FIRE!!!!!!!!][Safe.....] |
| Scenario_C_Slow.mp4  |  500  |   10%   |      0%       |     85%      |      10%       |     5%   | PRED: [Processed......][Processed...][Processed] |
|                      |       |         |    (Safe)     | (Inefficient)|                |          |       ^ WASTE (Processed Safe Frames)           |
+----------------------+-------+---------+---------------+--------------+----------------+----------+-------------------------------------------------+

# Legend for Your Report

Miss (FN): Fire was present, but you Skipped it. (Safety Failure)

Waste (FP): No fire was present, but you Processed it. (Efficiency Failure)

True Skip (TN): No fire was present, and you Skipped it. (Efficiency Success)

Hit (TP): Fire was present, and you Processed it. (Safety Success)