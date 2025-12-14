library ieee;
use ieee.std_logic_1164.all;

entity tb_ff is
end entity;

architecture sim of tb_ff is

    signal clk   : std_logic := '0';
    signal rst_a : std_logic := '0';
    signal rst_s : std_logic := '0';
    signal d     : std_logic := '0';
    signal q     : std_logic;

begin

    UUT: entity work.flipflop
        port map (
            clk   => clk,
            rst_a => rst_a,
            rst_s => rst_s,
            d     => d,
            q     => q
        );

    clk <= not clk after 10 ns;

    process
    begin
        rst_a <= '1';
        wait for 20 ns;
        rst_a <= '0';

        d <= '1';
        wait for 40 ns;

        rst_s <= '1';
        wait for 20 ns;
        rst_s <= '0';

        d <= '0';
        wait for 30 ns;

        wait;
    end process;

end architecture;
