#! /usr/bin/env python
# -*- coding: utf-8 -*-
#vim:fileencoding=utf-8
import os
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from leafCheck import leafCheck
from processLeaf import process
from classifyLeaf import classify

import random

def help(bot, update):
    a = """
    /start - start a chat
    /help - list of existing commands
    /trees_list - tree species I recognize
    If the leaf is compound (consists of several leaflets), take a picture of the top leaflet, please
    """
    update.message.reply_text(a)

def start(bot, update):
    update.message.reply_text('Send me a picture of a leaf, please. I will use it to determine a tree species :)')

def get_image(bot, update):
    file_id = update.message.photo[-1].file_id
    photo = bot.getFile(file_id)
    photo.download(file_id+'.png')
    checkedImage,cnt,coord = leafCheck(file_id+'.png')
    os.remove(file_id+'.png')
    if type(checkedImage) != str:
        update.message.reply_text(random.choice([
            'Nice photo!',
            "One moment, I'll check what tree is that",
            'Alright, processing...',
        ]))
        features = process(checkedImage,cnt,coord)
        result1, result2, result3 = classify(features)
        if result3 == 0:
            if result2 == 0:
                keyboard = [[InlineKeyboardButton(result1.capitalize(), 
                                                  callback_data=result1)]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                update.message.reply_text('This is most probably ' + result1 +
                                          '. More info on it:', reply_markup=reply_markup)
            else:
                keyboard = [[InlineKeyboardButton(result1.capitalize(),
                                                  callback_data=result1)],
                [InlineKeyboardButton(result2.capitalize(),
                                      callback_data=result2)]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                update.message.reply_text('This is either ' + result1 + ' or ' + 
                                      result2 + '. More details:', 
                                      reply_markup=reply_markup)
        else:
            keyboard = [[InlineKeyboardButton(result1.capitalize(),
                                              callback_data=result1)],
                 [InlineKeyboardButton(result2.capitalize(),
                                      callback_data=result2)],
                 [InlineKeyboardButton(result3.capitalize(),
                                      callback_data=result3)]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            update.message.reply_text('It looks like ' + result1 + ' or '
                                          + result2 + '. But it might also be ' +
                                          result3 + "! Read more:", reply_markup=reply_markup)
    else:
        update.message.reply_text(checkedImage)

def reply_text(bot, update):
    update.message.reply_text(random.choice([
         'Make sure there is only one leaf on a picture',
         'Want to know which tree is next to you?',
         'The weather is great, time to go to the park!',
         'Please send me a photo of the leaf'
    ]))

def trees_list(bot, update):
    myfile = open("trees.txt")
    msg = myfile.read()
    myfile.close()
    keyboard = map(create_button, msg.split('\n'))
    keyboard = keyboard[1:20]
    reply_markup = InlineKeyboardMarkup(keyboard)

    update.message.reply_text(u'List of trees:', reply_markup=reply_markup)

def create_button(name):
    return InlineKeyboardButton(name.capitalize(), callback_data = name),

def on_press_button(bot, update):
    query = update.callback_query

    myfile = open(u"trees/" + query.data + ".txt")
    msg = myfile.read()
    myfile.close()

    bot.edit_message_text(text=msg,
                          chat_id=query.message.chat_id,
                          message_id=query.message.message_id)

#def get_files(bot, update):
#    msg = os.listdir('/home/ifmoadmin')
#    update.message.reply_text(msg)

def main():
    updater = Updater('545225881:AAElIAyqmY6P_DYExioLMO3r6fkgC7N-KkQ')
    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('help',help))
    updater.dispatcher.add_handler(CallbackQueryHandler(on_press_button))
#    updater.dispatcher.add_handler(CommandHandler('get_files', get_files))
    updater.dispatcher.add_handler(CommandHandler('trees_list', trees_list))
    updater.dispatcher.add_handler(MessageHandler(filters.Filters.photo, get_image))
    updater.dispatcher.add_handler(MessageHandler(filters.Filters.text, reply_text))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
