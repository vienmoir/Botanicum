import os
from telegram.ext import Updater, CommandHandler, MessageHandler, filters
from telegram import InputFile

def help(bot, update):
    a = """
    /start - start chatting with me
    /help - list all available commands
    /get_files - info about all files on the server
    /trees_list - the trees I'm familiar with
    """
    update.message.reply_text(a)

def start(bot, update):
    update.message.reply_text('Send me a picture of a leaf. There should be only one leaf on a light background (not working for now).')

def get_image(bot, update):
    file_id = update.message.photo[-1].file_id
    photo = bot.getFile(file_id)
    photo.download(file_id+'.png')
    update.message.reply_text('Processing...')

def trees_list(bot, update):
    myfile = open("trees.txt")
    msg = myfile.read()
    myfile.close()
    update.message.reply_text(msg)

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
    updater.dispatcher.add_handler(CommandHandler('get_files', get_files))
    updater.dispatcher.add_handler(CommandHandler('trees_list', trees_list))
    updater.dispatcher.add_handler(MessageHandler(filters.Filters.photo, get_image))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
                                                              60,1          Bot
