import argparse

import cherrypy

from sigtestv.database import ResultsDatabase
from sigtestv.net import DatabaseLoggingService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database-file', '-d', type=str, required=True)
    parser.add_argument('--host', '-H', type=str, default='0.0.0.0')
    parser.add_argument('--port', '-p', type=int, default=5358)
    args = parser.parse_args()

    database = ResultsDatabase(args.database_file)
    conf = {'/': {
        'request.dispatch': cherrypy.dispatch.MethodDispatcher()
    }}
    cherrypy.tree.mount(DatabaseLoggingService(database), '/submit', conf)
    cherrypy.config.update({'server.socket_host': args.host,
                            'server.socket_port': args.port,
                            'environment': 'production'})
    cherrypy.engine.start()


if __name__ == '__main__':
    main()
