import telebot

Token = "6991697056:AAGwBuc9l1hY4l_YFVHBLd60nMMca3kuBoA"

bot = telebot.TeleBot(Token)

@bot.message_handler(['start'])
def start(message):
    bot.reply_to(message,"Welcome to Cnc_jugad")

@bot.message_handler(['help'])
def help(message):
    bot.reply_to(message,"This bot is help for convert 2d image to 3d image")

bot.polling()
