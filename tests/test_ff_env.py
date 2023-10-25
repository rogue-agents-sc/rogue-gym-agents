"""test for FirstFloorEnv and config string"""
from rogue_gym.envs import ImageSetting, FirstFloorEnv, StatusFlag
from rogue_gym.envs import FirstFloorEnv, ImageSetting, RogueEnv, StatusFlag
from data import CMD_STR2, SEED1_DUNGEON_CLEAR
import matplotlib.pyplot as plt

CONFIG = {
    "seed": 1,
    "hide_dungeon": False,
    "enemies": {
        "enemies": [],
    },
}

EXPAND = ImageSetting(status=StatusFlag.DUNGEON_LEVEL)


def test_configs():
    env = FirstFloorEnv(RogueEnv(config_dict=CONFIG, image_setting=EXPAND), 100.0)
    assert env.unwrapped.get_dungeon().__len__() == SEED1_DUNGEON_CLEAR.__len__()

    status = StatusFlag.EMPTY

    state, rewards, done, _ = env.step(CMD_STR2)
    gray_img = status.gray_image(state).squeeze()
    plt.imshow(
        gray_img, cmap="gray", extent=[0, gray_img.shape[1], 0, gray_img.shape[0]]
    )
    plt.show()

    assert done
    assert rewards == 102
    symbol_img = env.unwrapped.state_to_image(state)
    assert symbol_img.shape == (18, 24, 80)
    assert env.unwrapped.get_config() == CONFIG


if __name__ == "__main__":
    test_configs()
