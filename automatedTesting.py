"""
Title: Automated Testing
Author: Conor Cohen Farrell
Last Edited: 27 March 2019
GitHub: https://github.com/Ziyphex
Description: Automated Testing for Face Find App
"""

import pyautogui
screenWidth, screenHeight = pyautogui.size()
currentMouseX, currentMouseY = pyautogui.position()


# test that the help button acts as it should
pyautogui.moveTo(1656, 449)
pyautogui.click()
pyautogui.PAUSE = 1
pyautogui.moveTo(502, 113)
pyautogui.click()
pyautogui.PAUSE = 1


# test that the tabs are clickable and cleared
pyautogui.moveTo(1180, 485)
pyautogui.click()
pyautogui.PAUSE = 1
pyautogui.moveTo(1304, 482)
pyautogui.click()
pyautogui.PAUSE = 1
pyautogui.moveTo(1422, 484)
pyautogui.click()
pyautogui.PAUSE = 1
pyautogui.moveTo(1060, 484)
pyautogui.click()
pyautogui.PAUSE = 1


# test that the recoring works with live camera inactive
# click record button to start session
pyautogui.moveTo(1291, 916)
pyautogui.click()
pyautogui.PAUSE = 1
# observe each tab (the live camera should not open)
pyautogui.moveTo(1180, 485)
pyautogui.click()
pyautogui.PAUSE = 1
pyautogui.moveTo(1304, 482)
pyautogui.click()
pyautogui.PAUSE = 1
pyautogui.moveTo(1422, 484)
pyautogui.click()
pyautogui.PAUSE = 1
pyautogui.moveTo(1060, 484)
pyautogui.click()
pyautogui.PAUSE = 1
# click record button to stop session
pyautogui.moveTo(1291, 916)
pyautogui.click()
pyautogui.PAUSE = 1
# acknowledge popup and close it
pyautogui.moveTo(1480, 720)
pyautogui.click()
pyautogui.PAUSE = 1


# test that the recoring works with live camera active
# toggle checkbox to keep live camera open
pyautogui.moveTo(1237, 958)
pyautogui.click()
pyautogui.PAUSE = 1
# click record button to start session
pyautogui.moveTo(1291, 916)
pyautogui.click()
pyautogui.PAUSE = 1
# observe each tab (the live camera should not open)
pyautogui.moveTo(1180, 485)
pyautogui.click()
pyautogui.PAUSE = 1
pyautogui.moveTo(1304, 482)
pyautogui.click()
pyautogui.PAUSE = 1
pyautogui.moveTo(1422, 484)
pyautogui.click()
pyautogui.PAUSE = 1
pyautogui.moveTo(1060, 484)
pyautogui.click()
pyautogui.PAUSE = 1
# click record button to stop session
pyautogui.moveTo(1291, 916)
pyautogui.click()
pyautogui.PAUSE = 1
# acknowledge popup and close it
pyautogui.moveTo(1480, 720)
pyautogui.click()
pyautogui.PAUSE = 1
# toggle checkbox again to reset
pyautogui.moveTo(1237, 958)
pyautogui.click()
pyautogui.PAUSE = 1
