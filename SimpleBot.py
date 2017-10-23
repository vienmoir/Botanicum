import os
from telegram.ext import Updater, CommandHandler

def help(bot, update):
    a = """
    /start - start chatting with me
    /help - list all available commands
    /get_files - info about all files on the server
    /trees_list - the trees I'm familiar with
    /get_image - test function for images
    """
    update.message.reply_text(a)

def start(bot, update):
    update.message.reply_text('Send me the picture of a leaf. There should be only one leaf on a light background (not working for now).')

def get_image(bot, update):
    file_id = message.photo[2].file_id
    path = file_id+'.jpg'
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(path,'wb') as new_file:
        new_file.write(downloaded_file)
    #newImage = bot.getFile(file_id)
    #newImage.download('voice.ogg')
    update.message.photo[-1](path)

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
    updater.dispatcher.add_handler(CommandHandler('get_image', get_image))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
                 