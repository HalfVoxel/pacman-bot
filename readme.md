Bot for the Pacman Tournament
=============================

http://ai.berkeley.edu/contest.html

Usage

~~~
# Clone the tournament repo and this repo
git clone git@gits-15.sys.kth.se:antthu/pacman-tournament.git
git clone <this repository>
# Make a symbolic link to the bot for this group
cd pacman-tournament/teams
ln -s ../../pacman-bot/bot.py group-4.py
cd ..
# Run the game (with this bot as the red team)
python3 capture.py -r teams/group-4.py