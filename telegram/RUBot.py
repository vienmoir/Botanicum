# -*- coding: utf-8 -*-
import os
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from leafCheck import leafCheck
from processLeaf import process
from RUclassifyLeaf import classify

import random

def help(bot, update):
    a = """
    /start - начать чат
    /help - список существующих команд
    /trees_list - известные мне деревья
    Если лист дерева сложный (состоит из нескольких маленьких), сфотографируйте только верхний листик, пожалуйста
    """
    update.message.reply_text(a)

def start(bot, update):
    update.message.reply_text('Отправьте мне фото листа на светлом нейтральном фоне. Я постараюсь определить, какому дереву он принадлежал :)')

def get_image(bot, update):
    file_id = update.message.photo[-1].file_id
    photo = bot.getFile(file_id)
    photo.download(file_id+'.png')
    checkedImage,cnt,coord = leafCheck(file_id+'.png')
    if type(checkedImage) != str:
        update.message.reply_text(random.choice([
            'Отличное фото!',
            'Минутку, посмотрю в справочнике',
            'Всё в порядке, обрабатываю',
        ]))
        features = process(checkedImage,cnt,coord)
        result1, result2, result3 = classify(features)
        if result3 == 0:
            if result2 == 0:
                keyboard = [[InlineKeyboardButton(result1.decode('utf-8').capitalize(), 
                                                  callback_data=result1)]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                update.message.reply_text('Скорее всего, это ' + result1 +
                                          '. Подробнее об этом виде:', reply_markup=reply_markup)
            else:
                keyboard = [[InlineKeyboardButton(result1.decode('utf-8').capitalize(),
                                                  callback_data=result1)],
                [InlineKeyboardButton(result2.decode('utf-8').capitalize(),
                                      callback_data=result2)]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                update.message.reply_text('Похоже, это ' + result1 + ' или ' + 
                                      result2 + '. Вот их описания:', 
                                      reply_markup=reply_markup)
        else:
            keyboard = [[InlineKeyboardButton(result1.decode('utf-8').capitalize(),
                                              callback_data=result1)],
                 [InlineKeyboardButton(result2.decode('utf-8').capitalize(),
                                      callback_data=result2)],
                 [InlineKeyboardButton(result3.decode('utf-8').capitalize(),
                                      callback_data=result3)]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            update.message.reply_text('Кажется, это ' + result1 + ' или '
                                          + result2 + '. Но может быть и ' +
                                          result3 + "! Подробнее о них:", reply_markup=reply_markup)
    else:
        update.message.reply_text(checkedImage)

def reply_text(bot, update):
    update.message.reply_text(random.choice([
        'Следите, чтобы пальцы не попали в кадр',
        'Хотите узнать, какое рядом с вами дерево?',
        'Погода отличная, пора в парк!',
        'Пожалуйста, отправьте мне фото листика'
    ]))

def trees_list(bot, update):
    myfile = open("trees.txt")
    msg = myfile.read()
    myfile.close()
    keyboard = map(create_button, msg.split('\n'))
    keyboard = keyboard[1:20]
    reply_markup = InlineKeyboardMarkup(keyboard)

    update.message.reply_text('Виды деревьев:', reply_markup=reply_markup)

def create_button(name):
    return InlineKeyboardButton(name.decode('utf-8').capitalize(), callback_data = name),

def on_press_button(bot, update):
    query = update.callback_query

    myfile = open(u"trees/" + query.data + ".txt")
    msg = myfile.read()
    myfile.close()

    bot.edit_message_text(text=msg,
                          chat_id=query.message.chat_id,
                          message_id=query.message.message_id)

def get_files(bot, update):
    msg = os.listdir('/home/ifmoadmin')
    update.message.reply_text(msg)

def main():
    updater = Updater('459746778:AAHDw1iCbP_FBslNlica-NxYQ02c3ZsmJ4Q')
    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('help',help))
    updater.dispatcher.add_handler(CallbackQueryHandler(on_press_button))
    updater.dispatcher.add_handler(CommandHandler('get_files', get_files))
    updater.dispatcher.add_handler(CommandHandler('trees_list', trees_list))
    updater.dispatcher.add_handler(MessageHandler(filters.Filters.photo, get_image))
    updater.dispatcher.add_handler(MessageHandler(filters.Filters.text, reply_text))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
