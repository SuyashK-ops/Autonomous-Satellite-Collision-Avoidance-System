#The first function is a very simplified maneuver:


from poliastro.twobody import Orbit

def apply_maneuver(orbit, delta_v):

    r = orbit.r
    v = orbit.v + delta_v

    new_orbit = Orbit.from_vectors(
        orbit.attractor,
        r,
        v,
        epoch=orbit.epoch
    )

    return new_orbit

#Later we will compute optimal avoidance burns.