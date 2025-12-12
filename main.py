from cli import Cli


def main():
    cli = Cli()
    while True:
        try:
            cli.run_monadic()
        except KeyboardInterrupt:
            print("User interrupted the execution.")
            print("Goodbye!")
            break
        except Exception as e:
            print(e.with_traceback(None))


if __name__ == '__main__':
    main()
