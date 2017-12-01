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
    """
    update.message.reply_text(a)

def start(bot, update):
    #update.message.reply_text('Send me the picture of a leaf. There should be only one leaf on a light background (not working for now).')
    update.message.reply_text('Отправьте мне фото листа на светлом нейтральном фоне. Я постараюсь определить, какому дереву он принадлежал :)')

def get_image(bot, update):
    file_id = update.message.photo[-1].file_id
    photo = bot.getFile(file_id)
    photo.download(file_id+'.png')
    print "yes"
    checkedImage,cnt,coord = leafCheck(file_id+'.png')
    print "yeah"
    if type(checkedImage) != str:
        print "yass"
        update.message.reply_text(random.choice([
            'Вот так лист!',
            'Отличное фото!',
            'Минутку, посмотрю в справочнике',
            'Всё в порядке, обрабатываю',
        ]))
        features = process(checkedImage,cnt,coord)
        print "yup"
        result1, result2, result3 = classify(features)
        print "yep"
        if result3 == 0:
            if result2 == 0:
                update.message.reply_text("Скорее всего, это " + result1 +
                                          ". Подробнее об этом виде:")
            else:
                update.message.reply_text("Похоже, это " + result1 + " или " + 
                                      result2 + ". Вот их описания:")
        else:
            update.message.reply_text("Кажется, это " + result1 + " или "
                                          + result2 + ". Но может быть и " +
                                          result3 + "! Подробнее о них:")
        update.message.reply_text(result)
    else:
        update.message.reply_text(checkedImage)

def reply_text(bot, update):
    update.message.reply_text(random.choice([
        'Следите, чтобы пальцы не попали в кадр',
        'Хотите узнать, какое рядом с вами дерево?',
        'Погода отличная, пора в парк!',
        'Пожалуйста, отправьте мне фото листика',
    ]))

def trees_list(bot, update):
    myfile = open("trees.txt")
    msg = myfile.read()
    myfile.close()
    keyboard = map(create_button, msg.split('\n'))
    reply_markup = InlineKeyboardMarkup(keyboard)

    update.message.reply_text('Виды деревьев:', reply_markup=reply_markup)

def create_button(name):
    return InlineKeyboardButton(name, callback_data=name.split('.')[0]),


def on_press_button(bot, update):
    query = update.callback_query

    myfile = open("trees/" + query.data + ".txt")
    msg = myfile.read()
    myfile.close()

    bot.edit_message_text(text=msg,
                          chat_id=query.message.chat_id,
                          message_id=query.message.message_id)

def get_files(bot, update):
    #if update.message.chat.username == None:
        #update.message.reply_text('Add username to telegram account please')
        #return
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
