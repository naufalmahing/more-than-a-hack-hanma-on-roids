# The Approach
We use Rule Based Possession Model, since training data is a constraint and we can infer directly from dataset. 

We detect the similarity of movement from ball and player and also find the nearest player to ball distance.

# Assets
We use timestamp, frame, ball speed, ball position, player speed, player position

# Training Data
We don't do any training, we infer from the match directly

# Possession Rules
We drew the line of possession with a smoothing window. Detect the nearest player and similarity of movement and calculate which team has a better score within a window. We label every frame based on who is the nearest and has the most similar movement to the ball. We don't care about the event that is happening.

For restarts, so long as the ball is in the pitch, we know is on play.

# Limitations
The model can't detect deadball, halftime, and injury stoppages. We would add another rule or method to infer deadball and halftime.

# Reproduction
We convert the BAR-ATH_tactical_atd.xml to csv, and infer the result from it.

run possession_detector.py "data/tactical/compiled_match.csv" --output predictions.csv

Extract the frame and label from output file and put it in evaluation/results/
