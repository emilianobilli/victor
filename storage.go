package victor

type Storage interface {
	save(*Record) error
	load(string) (*Record, error)
	check(string) bool
	delete(string) error
	list() ([]string, error)
}
