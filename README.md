# GaryBot
A vibe code Robocode Tank Royale Bot

## Running the bot

Place the `GaryBot` folder in your Robocode Tank Royale bots directory
(next to the sample bots, which provide the shared `deps` folder).

## GaryTeam

`team/GaryTeam/GaryTeam.json` defines a team of four GaryBots.
To use it, copy the `GaryTeam` folder into your bots directory, so it sits
next to the `GaryBot` folder:

```
bots/
├── deps/
├── GaryBot/
└── GaryTeam/
    └── GaryTeam.json
```

GaryBot's `team_mode` flag (on by default) makes teammates never target
each other and hold fire when a teammate is in the line of fire.
