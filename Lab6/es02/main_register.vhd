library ieee;
use ieee.std_logic_1164.all;

entity main_register is
    generic(
        N : integer := 8       -- numero di bit del registro
    );
    port(
        clk   : in std_logic;
        rst_a : in std_logic;
        rst_s : in std_logic;
        d     : in std_logic_vector(N-1 downto 0);
        q     : out std_logic_vector(N-1 downto 0)
    );
end entity;

architecture structural of main_register is
begin

    gen_reg : for i in 0 to N-1 generate
        FF_i : entity work.flipflop
            port map (
                clk   => clk,
                rst_a => rst_a,
                rst_s => rst_s,
                d     => d(i),
                q     => q(i)
            );
    end generate;

end architecture;
