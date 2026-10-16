import math
import random
from dataclasses import dataclass

from robocode_tank_royale.bot_api.bot import Bot
from robocode_tank_royale.bot_api.color import Color
from robocode_tank_royale.bot_api.constants import BOUNDING_CIRCLE_RADIUS, MAX_RADAR_TURN_RATE
from robocode_tank_royale.bot_api.events import (
    BotDeathEvent,
    BulletHitBotEvent,
    HitBotEvent,
    HitByBulletEvent,
    HitWallEvent,
    ScannedBotEvent,
)


# ------------------------------------------------------------------
# GaryBot
# ------------------------------------------------------------------
# A small but strong bot. Strategy in a nutshell:
#
#   Radar    : 1v1 -> lock onto the enemy ("wide lock").
#              melee -> spin all the time, so we keep seeing everybody.
#   Gun      : linear targeting - shoot where the enemy WILL be, not where it is.
#   Movement : drive perpendicular to the enemy, reverse when it fires
#              (detected by its energy drop) and slide along walls.
#   Team     : optional (team_mode flag) - ignore teammates as targets and
#              hold fire while one stands in the line of fire.
#
# Angle cheat sheet for Tank Royale (differs from classic Robocode!):
#   - 0 deg = east (+x), 90 deg = north (+y). Angles grow counter-clockwise.
#   - A positive *bearing* means "to the left", so turn_left(bearing) faces it.
#   - Moving in direction d with speed v: dx = cos(d) * v, dy = sin(d) * v
# ------------------------------------------------------------------

# Keep this far away from the walls (bot radius 18 + some safety).
WALL_MARGIN = 50
# How far ahead we "feel" for walls. Bigger = turns away earlier.
WALL_STICK = 140
# Distance we like to keep to the enemy. Close enough to hit, far enough to dodge.
PREFERRED_DISTANCE = 400
# Do not shoot at data older than this (turns). Old data = bad prediction.
MAX_DATA_AGE = 8
# Forget bots we have not seen for this many turns.
FORGET_AGE = 30
# Team mode: hold fire if a teammate is closer than this to our line of fire.
# Bot radius (18) + extra, because the teammate keeps moving while the bullet flies.
TEAMMATE_SAFETY = 50


@dataclass
class BotInfo:
    """What we know about one bot (enemy or teammate) from its last scan."""

    x: float
    y: float
    direction: float
    speed: float
    energy: float
    seen_turn: int


class GaryBot(Bot):
    def __init__(self) -> None:
        super().__init__()
        # True: when GaryBot fights as part of a team (e.g. GaryTeam), teammates are never
        # targeted and we hold fire when one is in the line of fire.
        # Safe to leave on for solo battles: is_teammate() is then False for every bot.
        # False: every scanned bot counts as an enemy, even teammates.
        self.team_mode: bool = True

        # All state lives on the instance and is reset at the start of every round in run().
        self._enemies: dict[int, BotInfo] = {}
        self._teammates: dict[int, BotInfo] = {}
        self._target_id: int | None = None
        self._move_dir: int = 1  # +1 = forward, -1 = backward
        self._next_random_flip: int = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Called when a new round is started -> set up, then act once per turn."""
        self._enemies.clear()
        self._teammates.clear()
        self._target_id = None
        self._move_dir = 1
        self._next_random_flip = 0

        self.body_color = Color.from_rgb(0x20, 0x40, 0x80)
        self.turret_color = Color.from_rgb(0xFF, 0xA0, 0x00)
        self.radar_color = Color.from_rgb(0xFF, 0xFF, 0xFF)
        self.bullet_color = Color.from_rgb(0xFF, 0x50, 0x00)
        self.scan_color = Color.from_rgb(0x80, 0xFF, 0x80)

        # Let body, gun and radar turn independently of each other.
        # Without this, turning the body would also drag the gun and radar
        # away from the enemy, and we could never move and aim at the same time.
        self.adjust_gun_for_body_turn = True
        self.adjust_radar_for_body_turn = True
        self.adjust_radar_for_gun_turn = True

        # Why not forward(100) / turn_gun_left(360) like MyFirstBot?
        # Those calls BLOCK: the bot does nothing else until the move is finished.
        # Instead we only *schedule* actions with set_*() and then call go() once,
        # which executes everything (move + turn + gun + radar) in the same turn.
        while self.running:
            target = self._select_target()
            self._do_radar(target)
            self._do_gun(target)
            self._do_movement(target)
            self.go()

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
    def _select_target(self) -> BotInfo | None:
        """Pick the closest enemy we have recent information about."""
        now = self.turn_number
        # Drop bots we have not seen for a long time - their data is useless.
        for bots in (self._enemies, self._teammates):
            for bot_id in [i for i, b in bots.items() if now - b.seen_turn > FORGET_AGE]:
                del bots[bot_id]

        if not self._enemies:
            self._target_id = None
            return None

        # Closest enemy = easiest to hit and usually the most dangerous one.
        self._target_id = min(self._enemies, key=lambda i: self.distance_to(self._enemies[i].x, self._enemies[i].y))
        return self._enemies[self._target_id]

    # ------------------------------------------------------------------
    # Radar
    # ------------------------------------------------------------------
    def _do_radar(self, target: BotInfo | None) -> None:
        age = self.turn_number - target.seen_turn if target else 999

        # Melee, or we lost the target: spin the radar forever.
        # A full circle takes 8 turns (45 deg/turn), so every enemy gets rescanned regularly.
        if target is None or self.enemy_count > 1 or age > 2:
            self.set_turn_radar_left(float("inf"))
            return

        # 1v1: "wide lock". Turn the radar to the enemy and then overshoot by the same
        # amount again. The scan arc is the area the radar sweeps in one turn, so the
        # overshoot guarantees the beam passes over the enemy every turn, even if it moves.
        # A radar that stops exactly on the enemy would lose it as soon as it moves.
        bearing = self.radar_bearing_to(target.x, target.y)
        turn = bearing * 2
        turn = max(-MAX_RADAR_TURN_RATE, min(MAX_RADAR_TURN_RATE, turn))
        # Bearing 0 would give no sweep (and no scan): always sweep at least a little.
        if abs(turn) < 5:
            turn = 5 if turn >= 0 else -5
        self.set_turn_radar_left(turn)

    # ------------------------------------------------------------------
    # Gun
    # ------------------------------------------------------------------
    def _do_gun(self, target: BotInfo | None) -> None:
        if target is None:
            return
        age = self.turn_number - target.seen_turn
        distance = self.distance_to(target.x, target.y)
        power = self._choose_firepower(target, distance)

        # Aim at the predicted position (bullet speed depends on firepower: 20 - 3 * power).
        aim_x, aim_y = self._predict_position(target, self.calc_bullet_speed(power), age)
        gun_bearing = self.gun_bearing_to(aim_x, aim_y)

        # Only shoot if the gun ALREADY points at the aim point (within the width of the enemy).
        # The gun can turn only 20 deg/turn, so right after a big target change it is not there yet.
        # Firing anyway would waste energy on a sure miss and heat up the gun.
        tolerance = math.degrees(math.atan2(BOUNDING_CIRCLE_RADIUS, max(distance, 1)))
        if (
            self.gun_heat == 0
            and abs(gun_bearing) <= tolerance
            and age <= MAX_DATA_AGE
            and power >= 0.1
            and not self._teammate_in_line_of_fire(aim_x, aim_y)
        ):
            self.set_fire(power)

        self.set_turn_gun_left(gun_bearing)

    def _teammate_in_line_of_fire(self, aim_x: float, aim_y: float) -> bool:
        """Team mode: is a teammate close to the straight line from us to the aim point?

        Bullets fly in a straight line and hit the FIRST bot in their way, friend or foe.
        So we check the distance from each teammate to the line segment us -> aim point.
        """
        if not self.team_mode:
            return False
        seg_x, seg_y = aim_x - self.x, aim_y - self.y
        seg_len_sq = seg_x * seg_x + seg_y * seg_y
        if seg_len_sq == 0:
            return False
        for mate in self._teammates.values():
            # Project the teammate onto the line: t = 0 is us, t = 1 is the aim point.
            # Clamping to [0, 1] means teammates behind us or beyond the target are measured
            # to the nearest end of the segment, so they only block if they are really close.
            t = ((mate.x - self.x) * seg_x + (mate.y - self.y) * seg_y) / seg_len_sq
            t = max(0.0, min(1.0, t))
            closest_x, closest_y = self.x + t * seg_x, self.y + t * seg_y
            if math.hypot(mate.x - closest_x, mate.y - closest_y) < TEAMMATE_SAFETY:
                return True
        return False

    def _choose_firepower(self, target: BotInfo, distance: float) -> float:
        # Close: big bullets hit almost always. Far: small bullets fly faster
        # (so they are easier to land) and cost less energy when they miss.
        power = max(1.1, min(3.0, 600 / max(distance, 1)))

        # Never shoot more than needed to finish the enemy off.
        # Bullet damage = 4 * power, plus 2 * (power - 1) if power > 1.
        if target.energy <= 4:
            needed = target.energy / 4
        else:
            needed = (target.energy + 2) / 6
        power = min(power, needed + 0.1)

        # Firing costs our own energy. At 0 energy we are "disabled" (cannot move or shoot),
        # so when we are weak we shoot small and always keep a reserve.
        if self.energy < 20:
            power = min(power, self.energy / 10)
        return min(power, self.energy - 0.1)

    def _predict_position(self, target: BotInfo, bullet_speed: float, age: int) -> tuple[float, float]:
        """Linear targeting: assume the enemy keeps its direction and speed.

        We step forward in time until our bullet would have flown far enough to reach
        the enemy's position at that time. Iterating (instead of a single formula) is
        simple and lets us keep the predicted point inside the arena at every step.
        """
        dx = math.cos(math.radians(target.direction)) * target.speed
        dy = math.sin(math.radians(target.direction)) * target.speed
        min_x, min_y = BOUNDING_CIRCLE_RADIUS, BOUNDING_CIRCLE_RADIUS
        max_x = self.arena_width - BOUNDING_CIRCLE_RADIUS
        max_y = self.arena_height - BOUNDING_CIRCLE_RADIUS

        # Start from where the enemy probably is NOW (our scan may be a few turns old).
        x = max(min_x, min(max_x, target.x + dx * age))
        y = max(min_y, min(max_y, target.y + dy * age))

        t = 0
        while bullet_speed * t < self.distance_to(x, y) and t < 100:
            x = max(min_x, min(max_x, x + dx))
            y = max(min_y, min(max_y, y + dy))
            t += 1
        return x, y

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------
    def _do_movement(self, target: BotInfo | None) -> None:
        # Random direction changes so that we never move in a pattern that can be learned.
        if self.turn_number >= self._next_random_flip:
            if self._next_random_flip != 0:
                self._move_dir *= -1
            self._next_random_flip = self.turn_number + random.randint(20, 50)

        if target is None:
            # Nobody seen yet: head for the center, there we are safe from walls.
            travel_heading = self.direction_to(self.arena_width / 2, self.arena_height / 2)
        else:
            # Drive at 90 deg to the line between us and the enemy. Bullets fly straight,
            # so moving sideways makes us the hardest possible target.
            # Tilt slightly toward or away from the enemy to hold our preferred distance.
            distance = self.distance_to(target.x, target.y)
            tilt = max(-30.0, min(30.0, (distance - PREFERRED_DISTANCE) / 10))
            travel_heading = self.direction_to(target.x, target.y) - self._move_dir * (90 - tilt)

        travel_heading = self._wall_smooth(travel_heading)

        # We can drive forward or backward. Backward means the body points the other way.
        body_heading = travel_heading if self._move_dir > 0 else travel_heading + 180
        self.set_turn_left(self.calc_bearing(body_heading))
        # Re-issued every turn, so we never actually arrive and stop. Standing still = easy target.
        self.set_forward(100 * self._move_dir)

    def _wall_smooth(self, heading: float) -> float:
        """Rotate the heading (a bit more each try) until a point WALL_STICK ahead is inside the arena.

        This lets us slide along walls instead of crashing into them. Hitting a wall
        costs energy and stops us dead, which makes us easy to hit.
        """
        for step in range(36):
            # Try 0, +5, -5, +10, -10, ... degrees away from the wanted heading.
            delta = (step + 1) // 2 * 5 * (1 if step % 2 else -1)
            candidate = heading + delta
            x = self.x + math.cos(math.radians(candidate)) * WALL_STICK
            y = self.y + math.sin(math.radians(candidate)) * WALL_STICK
            if WALL_MARGIN < x < self.arena_width - WALL_MARGIN and WALL_MARGIN < y < self.arena_height - WALL_MARGIN:
                return candidate
        return self.direction_to(self.arena_width / 2, self.arena_height / 2)

    # ------------------------------------------------------------------
    # Events: these only update our knowledge or flip the movement direction.
    # No blocking calls here - the main loop does the actual work every turn.
    # ------------------------------------------------------------------
    def on_scanned_bot(self, e: ScannedBotEvent) -> None:
        """We saw another bot -> remember where it is and check whether it just fired."""
        # The radar sees teammates too. In team mode keep them in a separate list,
        # so they are never chosen as target but we still know where they are.
        if self.team_mode and self.is_teammate(e.scanned_bot_id):
            self._teammates[e.scanned_bot_id] = BotInfo(e.x, e.y, e.direction, e.speed, e.energy, self.turn_number)
            return

        old = self._enemies.get(e.scanned_bot_id)

        # Firing a bullet costs the shooter 0.1 - 3.0 energy. There is no "enemy fired"
        # event, but an energy drop in that range almost always means a bullet is coming.
        # Changing direction right then makes aiming bots (most samples) miss.
        if old is not None and e.scanned_bot_id == self._target_id:
            drop = old.energy - e.energy
            if 0.09 <= drop <= 3.01 and random.random() < 0.6:
                self._move_dir *= -1

        self._enemies[e.scanned_bot_id] = BotInfo(e.x, e.y, e.direction, e.speed, e.energy, self.turn_number)

    def on_bullet_hit(self, e: BulletHitBotEvent) -> None:
        """Our bullet hit -> update the enemy's energy, so the drop is not mistaken for it firing."""
        enemy = self._enemies.get(e.victim_id)
        if enemy is not None:
            enemy.energy = e.energy

    def on_hit_by_bullet(self, e: HitByBulletEvent) -> None:
        """We were hit -> whatever we are doing is predictable, so change direction."""
        del e
        self._move_dir *= -1

    def on_hit_wall(self, e: HitWallEvent) -> None:
        """Wall smoothing should prevent this, but if it happens: back off."""
        del e
        self._move_dir *= -1

    def on_hit_bot(self, e: HitBotEvent) -> None:
        """We bumped into a bot -> get away. Ramming hurts both bots."""
        del e
        self._move_dir *= -1

    def on_bot_death(self, e: BotDeathEvent) -> None:
        """A bot died -> forget it, so we do not keep aiming at a wreck."""
        self._enemies.pop(e.victim_id, None)
        self._teammates.pop(e.victim_id, None)


def main() -> None:
    bot = GaryBot()
    bot.start()


if __name__ == "__main__":
    main()
