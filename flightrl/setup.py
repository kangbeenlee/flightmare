import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

setup(
    name='rpg_baselines',
    version='0.0.1',
    author='Yunlong Song',
    author_email='song@ifi.uzh.ch',
    description='Flightmare: A Quadrotor Simulator.',
    long_description='',
    install_requires=['gym==0.11', 'ruamel.yaml', 'numpy', 'stable_baselines==2.10.1'],
    packages=['rpg_baselines',
              'rpg_baselines.envs',
              'rpg_baselines.single_agent',
              'rpg_baselines.single_agent.ddpg',
              'rpg_baselines.single_agent.td3',
              'rpg_baselines.single_agent.ppo',
              'rpg_baselines.multi_agent.maddpg',
              'rpg_baselines.multi_agent.matd3',
              'rpg_baselines.multi_agent.mappo'],
)
