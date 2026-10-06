# bluffjax.wHLO

Play all ten [BluffJAX](https://github.com/bluffjax/bluffjax) games in the
browser against bots, running the real BluffJAX JAX code client-side.

Each game's `reset`, `step_env` and `get_avail_actions` are exported with
`jax.export` to StableHLO and compiled to WebAssembly in the page by
[whlo](https://github.com/noahfarr/whlo), a StableHLO compiler that runs in
the browser. There is no server and no Python at play time: the game logic
you play against is the same JAX code you train on.

Games: Kuhn poker, Leduc hold'em, Goofspiel, Bluff, Kemps, Werewolf, 5-card
draw, 7-card stud, limit hold'em and no-limit hold'em.

## Run it

Everything needed is checked in (exported games plus a vendored whlo build),
so any static file server works:

```sh
python3 -m http.server 8000
# open http://localhost:8000/        (deep link to a game: /#werewolf)
```

You always sit in seat 0. The other seats are simple heuristic bots written
in `index.html` (hand-strength betting, impossible-claim challenges in Bluff,
accusation-following in Werewolf), not trained policies. Swapping in a
trained policy would mean exporting its `apply` the same way as the games.

## How it works

`export_games.py` flattens each env's state pytree into its leaves and
exports three functions per game to `games/<name>/`:

```
reset(key: u32[2])                  -> (*state_leaves)
step(key: u32[2], *leaves, action)  -> (*state_leaves, reward f32[num_agents], done i32[])
avail(*leaves)                      -> legal-action mask
```

`step` wraps `step_env` rather than `step`, so a finished hand stays on the
table until the page deals the next one. For simultaneous games (Goofspiel,
Kemps) `action` is a vector with one entry per agent.

`games/index.json` names every state leaf by its field path (`agent_cards`,
`pile_claims`, `roles`, ...). BluffJAX has no renderer, so `index.html`
draws each game straight from those named fields and hides what seat 0
shouldn't see: opponents' cards until showdown, Werewolf roles you don't
know.

## Re-export after changing BluffJAX

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --no-deps -e /path/to/bluffjax
uv pip install --python .venv/bin/python jax flax jaxtyping
JAX_PLATFORMS=cpu .venv/bin/python export_games.py            # all games, or: ... kuhn_poker bluff
```

## Checking it against JAX

```sh
.venv/bin/python verify/make_ref.py kuhn_poker 400   # JAX rollout with random legal actions
node verify/check.mjs kuhn_poker                     # replay through whlo and compare
```

`check.mjs` compares the full state, rewards, done flags and legal-action
masks at every step. All ten games match JAX bit for bit over 400 random
legal moves each (more than 1,000 hands in total).

## Credits

Game rules and dynamics are BluffJAX's; this repo only exports and renders
them. `vendor/whlo/` is a build of [whlo](https://github.com/noahfarr/whlo)
(Apache-2.0, see `vendor/whlo/LICENSE`).
