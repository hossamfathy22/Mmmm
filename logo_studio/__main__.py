"""python -m logo_studio  ->  http://localhost:8765"""
import os, webbrowser, threading
import uvicorn

def main():
    port = int(os.environ.get('PORT', 8765))
    if os.environ.get('LOGO_STUDIO_NO_BROWSER') != '1':
        threading.Timer(1.5, lambda: webbrowser.open('http://localhost:%d' % port)).start()
    uvicorn.run('logo_studio.server:app', host='127.0.0.1', port=port, log_level='warning')

if __name__ == '__main__':
    main()
